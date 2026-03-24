import pandas as pd
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb


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

TOURNAMENT_CONTEXT = """
TOURNAMENT: First Stand 2026 (São Paulo, Brazil, March 16-22 2026)
FORMAT: 8 teams, GSL double-elimination groups into knockout stage. All matches are Best of 5.

RESULTS:
- Finals: BLG 3-1 G2 (Finals MVP: Bin)
- Semi 1: G2 3-0 GenG
- Semi 2: BLG 3-0 JDG
- Group A: BLG and G2 advanced. BFX and TSW eliminated.
- Group B: GenG and JDG advanced. LYON and LOUD eliminated.

TEAM ROSTERS:
BLG (Bilibili Gaming, LPL, CHAMPIONS): Bin (Top), Xun (Jungle), knight (Mid), Viper (Bot), ON (Support). Coach: Dany
G2 Esports (LEC, Runners-up): BrokenBlade/BB (Top), SkewMond (Jungle), Caps (Mid), Hans Sama (Bot), Labrov (Support)
Gen.G (LCK #1): Kiin (Top), Canyon (Jungle), Chovy (Mid), Ruler (Bot), Duro (Support)
BNK FearX/BFX (LCK #2): Clear (Top), Raptor (Jungle), VicLa (Mid), Diable (Bot), Kellin (Support)
JD Gaming/JDG (LPL): Xiaoxu (Top), JunJia (Jungle), HongQ (Mid), GALA (Bot), Vampire (Support). Coach: Tabe
LYON (LCS): Dhokla (Top), Inspired (Jungle), Saint (Mid), Berserker (Bot), Isles (Support)
LOUD (CBLOL, host team): Xyno (Top), YoungJae (Jungle), Envy (Mid), Bull (Bot), RedBert (Support)
Team Secret Whales/TSW (LCP): Pun (Top), Hizto (Jungle), Dire (Mid), Eddie (Bot), Bie (Support)

NOTABLE CHAMPION STATS:
Most picked: Yunara (66.7%), Xin Zhao (70%), Vi (45.5%), Ahri (41.7%), Aurora (27.3%)
Most banned: Orianna (61%), Ryze (51.2%), Karma (26.8%), Rumble (24.4%), Varus (19.5%)
"""

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
    # Flip distances (smaller distance = better) and normalize
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

    # Get top results — take all comments with a hybrid score above 0.1 (at least somewhat relevant), max 20
    sorted_indices = np.argsort(hybrid_scores)[::-1]
    top_indices = [i for i in sorted_indices if hybrid_scores[i] > 0.1][:20]

    # Build context from results
    context = ""
    for rank, i in enumerate(top_indices, 1):
        print(f"\n--- Result {rank} ---")
        print(f"Match: {df.iloc[i]['match']}")
        print(f"Body: {df.iloc[i]['body'][:200]}")
        print(f"Hybrid: {hybrid_scores[i]:.3f}  (BM25: {bm25_normalized[i]:.3f}, Semantic: {semantic_scores[i]:.3f})")
        context += f"[Comment from {df.iloc[i]['match']}, {int(df.iloc[i]['score'])} upvotes]: {df.iloc[i]['body']}\n\n"

    print(f"\n({len(top_indices)} relevant comments found)")

    # Ask Ollama
    print("\nThinking...")
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": f"You are an esports analyst for League of Legends. Use the tournament info and community comments below to answer the question. Be concise and accurate.\n\n{TOURNAMENT_CONTEXT}\n\nRelevant community comments:\n{context}\nQuestion: {query}",
        "stream": False
    })
    print(f"\nAnswer: {response.json()['response']}")
