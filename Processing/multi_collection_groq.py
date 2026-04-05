"""
Multi-Collection RAG Pipeline for First Stand 2026 (Groq Cloud LLM)

Same as multi_collection_search.py but uses Groq API (llama-3.3-70b-versatile)
instead of a local Ollama instance. No local GPU required.

Usage: python3 multi_collection_groq.py
       (run from the Processing/ directory)
"""

import pandas as pd
import numpy as np
import requests
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

KNOWLEDGE_DIR = "../Knowledge"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# ──────────────────────────────────────────────────────────────
# 1. LOAD & CHUNK ALL DATA SOURCES
# ──────────────────────────────────────────────────────────────

def load_comment_chunks(csv_path="../GatheringData/cleaned_resonating_strike_data.csv"):
    """Load Reddit comments — each comment is one chunk."""
    df = pd.read_csv(csv_path)
    chunks = []
    for _, row in df.iterrows():
        chunks.append({
            "text": row["body"],
            "metadata": {
                "type": "comment",
                "source": "reddit",
                "match": row["match"],
                "upvotes": int(row["score"]),
            }
        })
    return df, chunks


def load_player_chunks(path=f"{KNOWLEDGE_DIR}/players.json"):
    """Load player profiles — one chunk per player."""
    with open(path, "r") as f:
        data = json.load(f)

    chunks = []
    for ign, player in data["players"].items():
        # Build a rich text paragraph for this player
        text = (
            f"{player['ign']} ({player['real_name']}) is a {player['role']} player for {player['team']}. "
            f"Nationality: {player['nationality']}. "
            f"Signature champions: {', '.join(player.get('signature_champions', []))}. "
            f"Playstyle: {player['playstyle']} "
            f"Tournament performance: {player['tournament_performance']} "
            f"Community sentiment: {player.get('community_sentiment', 'N/A')}"
        )
        chunks.append({
            "text": text,
            "metadata": {
                "type": "player",
                "source": "knowledge_base",
                "name": player["ign"],
                "team": player["team"],
                "role": player["role"],
            }
        })

    # Also add coach profiles
    for team, coach in data.get("coaches", {}).items():
        text = (
            f"Coach {coach['ign']} ({coach['real_name']}) coaches {team}. "
            f"Nationality: {coach['nationality']}. "
            f"Background: {coach['background']} "
            f"Tournament role: {coach.get('tournament_role', 'N/A')}"
        )
        chunks.append({
            "text": text,
            "metadata": {
                "type": "coach",
                "source": "knowledge_base",
                "name": coach["ign"],
                "team": team,
            }
        })

    return chunks


def load_champion_chunks(path=f"{KNOWLEDGE_DIR}/champions.json"):
    """Load champion data — one chunk per champion."""
    with open(path, "r") as f:
        data = json.load(f)

    chunks = []
    for name, champ in data["champions"].items():
        t = champ["tournament"]
        win_rate = f"{t['win_rate']}%" if t.get("win_rate") is not None else "N/A"
        context = champ.get("tournament_context", "")
        roles = t.get("roles", [t.get("primary_role", "Unknown")])
        role_str = "/".join(roles)

        text = (
            f"{champ['name']} ({champ.get('title', '')}): {', '.join(champ.get('tags', []))}. "
            f"Played: {role_str}. "
            f"Pick/Ban presence: {t['presence_pct']}%. "
            f"Games played: {t['games_played']}, Wins: {t['wins']}, Win rate: {win_rate}. "
            f"Banned {t['bans']} times. "
        )
        if context:
            text += f"Tournament context: {context}"

        chunks.append({
            "text": text,
            "metadata": {
                "type": "champion",
                "source": "data_dragon",
                "name": champ["name"],
                "role": t["primary_role"],
                "presence": t["presence_pct"],
            }
        })

    return chunks


def load_tournament_chunks(path=f"{KNOWLEDGE_DIR}/tournament.json"):
    """Load tournament data — one chunk per series/match + bracket info."""
    with open(path, "r") as f:
        data = json.load(f)

    chunks = []

    # Tournament overview chunk
    t = data["tournament"]
    overview = (
        f"Tournament: {t['name']}, held at {t['location']} from {t['dates']['start']} to {t['dates']['end']}. "
        f"Format: {t['format']}. "
        f"Champion: {t['champion']}. Runner-up: {t['runner_up']}."
    )
    chunks.append({
        "text": overview,
        "metadata": {"type": "tournament", "source": "official", "subtopic": "overview"}
    })

    # One chunk per match/series
    for group_name, group in data["bracket"]["group_stage"].items():
        for match in group["matches"]:
            text = (
                f"{match['stage']}: {match['team1']} vs {match['team2']}. "
                f"Winner: {match['winner']} ({match['score']}). "
                f"{match['summary']}"
            )
            # Add game-level details
            for game in match.get("games", []):
                if game.get("notable"):
                    text += f" Game {game['game']}: {game['notable']}."

            chunks.append({
                "text": text,
                "metadata": {
                    "type": "tournament",
                    "source": "official",
                    "subtopic": "match_result",
                    "match": match["match_label"],
                    "stage": match["stage"],
                }
            })

    # Semi-finals
    for semi in data["bracket"]["knockout_stage"]["semi_finals"]:
        text = (
            f"{semi['stage']}: {semi['team1']} vs {semi['team2']}. "
            f"Winner: {semi['winner']} ({semi['score']}). "
            f"{semi['summary']}"
        )
        for game in semi.get("games", []):
            if game.get("notable"):
                text += f" Game {game['game']}: {game['notable']}."
        for narrative in semi.get("narratives", []):
            text += f" {narrative}."

        chunks.append({
            "text": text,
            "metadata": {
                "type": "tournament",
                "source": "official",
                "subtopic": "match_result",
                "match": semi["match_label"],
                "stage": semi["stage"],
            }
        })

    # Finals
    finals = data["bracket"]["knockout_stage"]["finals"]
    text = (
        f"Grand Finals: {finals['team1']} vs {finals['team2']}. "
        f"Winner: {finals['winner']} ({finals['score']}). MVP: {finals['mvp']}. "
        f"{finals['summary']}"
    )
    for game in finals.get("games", []):
        if game.get("notable"):
            text += f" Game {game['game']}: {game['notable']}."
    for narrative in finals.get("narratives", []):
        text += f" {narrative}."

    chunks.append({
        "text": text,
        "metadata": {
            "type": "tournament",
            "source": "official",
            "subtopic": "match_result",
            "match": finals["match_label"],
            "stage": "Grand Finals",
        }
    })

    # Key narratives
    for narrative in data.get("key_narratives", []):
        chunks.append({
            "text": f"Tournament narrative — {narrative['narrative']}: {narrative['description']}",
            "metadata": {
                "type": "tournament",
                "source": "official",
                "subtopic": "narrative",
            }
        })

    # Team rosters overview
    for abbr, team in data["teams"].items():
        roster_parts = [f"{p['name']} ({role})" for role, p in team["roster"].items()]
        coach = team.get("coach")
        coach_str = f" Coach: {coach['ign']}" if isinstance(coach, dict) else ""
        text = (
            f"Team {team['full_name']} ({abbr}), {team['region']} {team['seed']}. "
            f"Result: {team['result']}. "
            f"Roster: {', '.join(roster_parts)}.{coach_str}"
        )
        chunks.append({
            "text": text,
            "metadata": {
                "type": "tournament",
                "source": "official",
                "subtopic": "team_roster",
                "team": abbr,
            }
        })

    return chunks


# ──────────────────────────────────────────────────────────────
# 2. BUILD CHROMADB COLLECTIONS
# ──────────────────────────────────────────────────────────────

def build_collection(client, model, name, chunks):
    """Create a ChromaDB collection from a list of chunks."""
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{name}_{i}" for i in range(len(chunks))]

    print(f"  Embedding {len(chunks)} chunks for '{name}'...")
    embeddings = model.encode(texts, show_progress_bar=True)

    collection = client.create_collection(name)
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids,
    )
    return collection


# ──────────────────────────────────────────────────────────────
# 3. MULTI-COLLECTION SEARCH
# ──────────────────────────────────────────────────────────────

def search_collection(collection, query, n_results=5):
    """Search a single ChromaDB collection and return results."""
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": doc,
            "metadata": meta,
            "distance": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def hybrid_comment_search(query, df, bm25, comments_collection, alpha=0.5, top_k=15):
    """Hybrid BM25 + semantic search for comments (existing logic preserved)."""
    # BM25 scores
    bm25_scores = bm25.get_scores(query.lower().split())
    max_bm25 = max(bm25_scores)
    bm25_normalized = bm25_scores / max_bm25 if max_bm25 > 0 else bm25_scores

    # Semantic distances
    results = comments_collection.query(
        query_texts=[query], n_results=len(df), include=["distances"]
    )
    distances = np.array(results["distances"][0])
    similarities = max(distances) - distances
    max_sim = max(similarities)
    semantic_normalized = similarities / max_sim if max_sim > 0 else similarities

    # Map back to original indices
    semantic_scores = np.zeros(len(df))
    for i, doc_id in enumerate(results["ids"][0]):
        idx = int(doc_id.replace("comments_", ""))
        semantic_scores[idx] = semantic_normalized[i]

    # Combine
    hybrid_scores = alpha * bm25_normalized + (1 - alpha) * semantic_scores
    sorted_indices = np.argsort(hybrid_scores)[::-1]
    top_indices = [i for i in sorted_indices if hybrid_scores[i] > 0.1][:top_k]

    return top_indices, hybrid_scores


def multi_collection_search(query, collections, df, bm25):
    """Search all collections and return structured results."""

    # 1. Comments: hybrid BM25 + semantic
    top_indices, hybrid_scores = hybrid_comment_search(
        query, df, bm25, collections["comments"]
    )
    comment_results = []
    for i in top_indices:
        comment_results.append({
            "text": df.iloc[i]["body"],
            "match": df.iloc[i]["match"],
            "upvotes": int(df.iloc[i]["score"]),
            "score": float(hybrid_scores[i]),
        })

    # 2. Players: semantic search
    player_results = search_collection(collections["players"], query, n_results=5)

    # 3. Champions: semantic search
    champion_results = search_collection(collections["champions"], query, n_results=5)

    # 4. Tournament: semantic search
    tournament_results = search_collection(collections["tournament"], query, n_results=5)

    return {
        "comments": comment_results,
        "players": player_results,
        "champions": champion_results,
        "tournament": tournament_results,
    }


# ──────────────────────────────────────────────────────────────
# 4. STRUCTURED PROMPT ASSEMBLY
# ──────────────────────────────────────────────────────────────

def build_prompt(query, results):
    """Build a natural-sounding prompt that blends all sources."""

    sections = []

    # Tournament info
    if results["tournament"]:
        tournament_text = "\n".join(
            f"- {r['text']}" for r in results["tournament"]
        )
        sections.append(f"Tournament info:\n{tournament_text}")

    # Player info
    if results["players"]:
        player_text = "\n".join(
            f"- {r['text']}" for r in results["players"]
        )
        sections.append(f"Player info:\n{player_text}")

    # Champion stats
    if results["champions"]:
        champion_text = "\n".join(
            f"- {r['text']}" for r in results["champions"]
        )
        sections.append(f"Champion stats:\n{champion_text}")

    # Community comments
    if results["comments"]:
        comment_text = "\n".join(
            f"[{c['match']}, {c['upvotes']} upvotes]: {c['text']}"
            for c in results["comments"][:15]
        )
        sections.append(f"Community reactions (Reddit):\n{comment_text}")

    context = "\n\n".join(sections)

    prompt = (
        "You are an esports analyst for League of Legends. You know the First Stand 2026 tournament inside out. "
        "Use the tournament info and community comments below to answer the question. "
        "Be concise and accurate. Blend facts with community sentiment naturally — "
        "don't be robotic about it, talk like someone who actually watches pro League.\n\n"
        f"{context}\n\n"
        f"Question: {query}"
    )

    return prompt


# ──────────────────────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("🔄 Loading data sources...")

    # Load all chunks
    df, comment_chunks = load_comment_chunks()
    player_chunks = load_player_chunks()
    champion_chunks = load_champion_chunks()
    tournament_chunks = load_tournament_chunks()

    print(f"   📝 Comments:   {len(comment_chunks)} chunks")
    print(f"   👤 Players:    {len(player_chunks)} chunks")
    print(f"   🎮 Champions:  {len(champion_chunks)} chunks")
    print(f"   🏆 Tournament: {len(tournament_chunks)} chunks")

    # Embedding model
    print("\n🧠 Loading embedding model...")
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    # Build collections
    print("\n📦 Initializing ChromaDB persistent storage...")
    client = chromadb.PersistentClient(path="./chroma_db")

    collections = {}
    datasets = {
        "comments": comment_chunks,
        "players": player_chunks,
        "champions": champion_chunks,
        "tournament": tournament_chunks,
    }

    for name, chunks in datasets.items():
        try:
            col = client.get_collection(name)
            if col.count() > 0 and col.count() == len(chunks):
                print(f"  ✅ Loaded existing collection '{name}' ({col.count()} chunks)")
                collections[name] = col
                continue
        except Exception:
            pass
        
        # If it doesn't exist or counts don't match (meaning data changed), rebuild
        try:
            client.delete_collection(name)
        except Exception:
            pass
        collections[name] = build_collection(client, model, name, chunks)

    # BM25 for comments (hybrid search)
    tokenized_comments = df["body"].str.lower().str.split()
    bm25 = BM25Okapi(tokenized_comments)

    print("\n✅ Multi-collection RAG pipeline ready! (Using Groq + llama-3.3-70b-versatile)")
    print("   Type a question, or 'quit' to exit.")
    print("   Type 'debug' before a query to see what was retrieved.\n")

    debug_mode = False

    while True:
        query = input("\nSearch: ").strip()
        if query.lower() == "quit":
            break

        if query.lower().startswith("debug"):
            debug_mode = not debug_mode
            print(f"   Debug mode: {'ON' if debug_mode else 'OFF'}")
            if query.lower() == "debug":
                continue
            query = query[5:].strip()

        if not query:
            continue

        # Search all collections
        results = multi_collection_search(query, collections, df, bm25)

        # Debug output
        if debug_mode:
            print(f"\n{'='*60}")
            print(f"🔍 RETRIEVAL DEBUG for: \"{query}\"")
            print(f"{'='*60}")
            print(f"\n🏆 Tournament results: {len(results['tournament'])}")
            for r in results["tournament"]:
                print(f"   [{r['distance']:.3f}] {r['text'][:100]}...")
            print(f"\n👤 Player results: {len(results['players'])}")
            for r in results["players"]:
                print(f"   [{r['distance']:.3f}] {r['metadata'].get('name', '?')} ({r['metadata'].get('team', '?')})")
            print(f"\n🎮 Champion results: {len(results['champions'])}")
            for r in results["champions"]:
                print(f"   [{r['distance']:.3f}] {r['metadata'].get('name', '?')} ({r['metadata'].get('role', '?')})")
            print(f"\n💬 Comment results: {len(results['comments'])}")
            for c in results["comments"][:5]:
                print(f"   [{c['score']:.3f}] [{c['match']}] {c['text'][:80]}...")
            print(f"{'='*60}\n")

        print(f"\n(Retrieved: {len(results['tournament'])} tournament, "
              f"{len(results['players'])} player, "
              f"{len(results['champions'])} champion, "
              f"{len(results['comments'])} comment chunks)")

        # Build structured prompt
        prompt = build_prompt(query, results)

        # Ask Groq
        print("\nThinking...")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            },
        )

        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            print(f"\nAnswer: {answer}")
        else:
            print(f"\n❌ Error {response.status_code}: {response.json()}")


if __name__ == "__main__":
    main()
