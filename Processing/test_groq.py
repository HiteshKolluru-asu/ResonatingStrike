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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def load_tournament_context(path="../Knowledge/tournament.json"):
    """Load structured tournament data and build a context string for the LLM."""
    with open(path, "r") as f:
        data = json.load(f)

    t = data["tournament"]
    lines = []
    lines.append(f"TOURNAMENT: {t['name']} ({t['location']}, {t['dates']['start']} to {t['dates']['end']})")
    lines.append(f"FORMAT: {t['format']}")
    lines.append(f"CHAMPION: {t['champion']} | RUNNER-UP: {t['runner_up']}")
    lines.append("")

    # Results
    lines.append("RESULTS:")
    finals = data["bracket"]["knockout_stage"]["finals"]
    lines.append(f"- Finals: {finals['team1']} {finals['score']} {finals['team2']} (Finals MVP: {finals['mvp']})")
    for semi in data["bracket"]["knockout_stage"]["semi_finals"]:
        lines.append(f"- {semi['stage']}: {semi['winner']} {semi['score']} (vs {semi['team1'] if semi['winner'] != semi['team1'] else semi['team2']})")
    for group_name, group in data["bracket"]["group_stage"].items():
        advanced = " and ".join(group["advanced"])
        eliminated = " and ".join(group["eliminated"])
        lines.append(f"- {group_name}: {advanced} advanced. {eliminated} eliminated.")
    lines.append("")

    # Team rosters
    lines.append("TEAM ROSTERS:")
    for abbr, team in data["teams"].items():
        roster_parts = [f"{player['name']} ({role})" for role, player in team["roster"].items()]
        roster_str = ", ".join(roster_parts)
        coach = team.get("coach")
        if isinstance(coach, dict):
            coach_str = f" Coach: {coach['ign']} ({coach['real_name']})"
        elif coach:
            coach_str = f" Coach: {coach}"
        else:
            coach_str = ""
        lines.append(f"{team['full_name']}/{abbr} ({team['region']}, {team['result']}): {roster_str}.{coach_str}")
    lines.append("")

    # Champion stats
    lines.append("NOTABLE CHAMPION STATS:")
    picked = ", ".join([f"{c['champion']} ({c['pick_rate']})" for c in data["champion_stats"]["most_picked"]])
    banned = ", ".join([f"{c['champion']} ({c['ban_rate']})" for c in data["champion_stats"]["most_banned"]])
    lines.append(f"Most picked: {picked}")
    lines.append(f"Most banned: {banned}")
    lines.append("")

    # Key narratives
    lines.append("KEY STORYLINES:")
    for n in data["key_narratives"]:
        lines.append(f"- {n['narrative']}: {n['description']}")

    return "\n".join(lines)


TOURNAMENT_CONTEXT = load_tournament_context()

df = pd.read_csv("../GatheringData/cleaned_resonating_strike_data.csv")

# BM25 setup
tokenized_comments = df['body'].str.lower().str.split()
bm25 = BM25Okapi(tokenized_comments)

# Semantic setup
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['body'].tolist(), show_progress_bar=True)

client = chromadb.Client()
collection = client.create_collection("comments")

collection.add(
    documents=df['body'].tolist(),
    embeddings=embeddings.tolist(),
    metadatas=[{"match": m, "score": int(s)} for m, s in zip(df['match'], df['score'])],
    ids=[str(i) for i in range(len(df))]
)

alpha = 0.5  # 0.5 = equal weight, higher = more BM25, lower = more semantic

print("✅ RAG pipeline ready! (Using Groq + llama-3.3-70b-versatile)\n")


while True:
    query = input("\nSearch: ")
    if query.lower() == 'quit':
        break

    # BM25 scores for ALL comments
    bm25_scores = bm25.get_scores(query.lower().split())
    max_bm25 = max(bm25_scores)
    if max_bm25 > 0:
        bm25_normalized = bm25_scores / max_bm25
    else:
        bm25_normalized = bm25_scores

    # Semantic distances for ALL comments
    results = collection.query(query_texts=[query], n_results=len(df), include=["distances"])
    distances = np.array(results['distances'][0])
    similarities = max(distances) - distances
    max_sim = max(similarities)
    if max_sim > 0:
        semantic_normalized = similarities / max_sim
    else:
        semantic_normalized = similarities

    # Map ChromaDB results back to original indices
    semantic_scores = np.zeros(len(df))
    for i, doc_id in enumerate(results['ids'][0]):
        semantic_scores[int(doc_id)] = semantic_normalized[i]

    # Combine
    hybrid_scores = alpha * bm25_normalized + (1 - alpha) * semantic_scores

    # Get top results
    sorted_indices = np.argsort(hybrid_scores)[::-1]
    top_indices = [i for i in sorted_indices if hybrid_scores[i] > 0.1][:20]

    # Build context from results
    context = ""
    for rank, i in enumerate(top_indices, 1):
        context += f"[Comment from {df.iloc[i]['match']}, {int(df.iloc[i]['score'])} upvotes]: {df.iloc[i]['body']}\n\n"

    print(f"\n({len(top_indices)} relevant comments found)")

    # Ask Groq (instead of Ollama)
    print("\nThinking...")
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "user",
                    "content": f"You are an esports analyst for League of Legends. Use the tournament info and community comments below to answer the question. Be concise and accurate.\n\n{TOURNAMENT_CONTEXT}\n\nRelevant community comments:\n{context}\nQuestion: {query}"
                }
            ],
            "temperature": 0.7
        }
    )

    if response.status_code == 200:
        answer = response.json()['choices'][0]['message']['content']
        print(f"\nAnswer: {answer}")
    else:
        print(f"\n❌ Error {response.status_code}: {response.json()}")
