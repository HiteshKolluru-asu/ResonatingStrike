import pandas as pd
from rank_bm25 import BM25Okapi


df = pd.read_csv("../GatheringData/cleaned_resonating_strike_data.csv")

tokenized_comments = df['body'].str.lower().str.split()

bm25 = BM25Okapi(tokenized_comments)

while True:
    query = input("\nSearch: ")
    if query.lower() == 'quit':
        break
    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    
    for rank, i in enumerate(top_n, 1):
        print(f"\n--- Result {rank} (BM25: {scores[i]:.2f}) ---")
        print(f"Match: {df.iloc[i]['match']}")
        print(f"Body: {df.iloc[i]['body'][:200]}")