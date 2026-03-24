from sentence_transformers import SentenceTransformer
import pandas as pd
import chromadb

model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("../GatheringData/cleaned_resonating_strike_data.csv")

embeddings = model.encode(df['body'].tolist(), show_progress_bar=True)


client = chromadb.Client()
collection = client.create_collection("comments")

collection.add(
    documents=df['body'].tolist(),
    embeddings=embeddings.tolist(),
    metadatas=[{"match": m, "score": int(s)} for m, s in zip(df['match'], df['score'])],
    ids=[str(i) for i in range(len(df))]
)

while True:
    query = input("\nSearch: ")
    if query.lower() == 'quit':
        break
    
    results = collection.query(query_texts=[query], n_results=5)
    
    for i in range(len(results['documents'][0])):
        print(f"\n--- Result {i+1} ---")
        print(f"Match: {results['metadatas'][0][i]['match']}")
        print(f"Body: {results['documents'][0][i][:200]}")