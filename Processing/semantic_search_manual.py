"""
Semantic search with manual embeddings + ChromaDB.
Opens up what SentenceTransformer('all-MiniLM-L6-v2').encode() does internally.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb

df = pd.read_csv("../GatheringData/cleaned_resonating_strike_data.csv")
print(f"Loaded {len(df)} comments\n")

# Load tokenizer and model separately (SentenceTransformer bundles these together)
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name)
transformer.eval()

print(f"Hidden size: {transformer.config.hidden_size}")            # 384-dim vectors
print(f"Layers: {transformer.config.num_hidden_layers}")           # L6 = 6 layers
print(f"Attention heads: {transformer.config.num_attention_heads}") # 12 parallel attention streams
print(f"Vocab size: {tokenizer.vocab_size}")                       # 30,522 known tokens


def embed_text(text):
    """
    What model.encode() does: Tokenize → Transformer → Mean Pool → Normalize
    """
    # Tokenize: text → token IDs + attention mask (1=real, 0=padding)
    encoded = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

    # Forward pass: 6 layers of self-attention refine each token's 384-dim representation
    with torch.no_grad():
        outputs = transformer(**encoded)

    # last_hidden_state: (1, num_tokens, 384) — one 384-dim vector per token
    token_embeddings = outputs.last_hidden_state

    # Mean pooling: average token vectors → single sentence vector (masking out padding)
    mask = encoded['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sentence_vector = (torch.sum(token_embeddings * mask, dim=1) / mask.sum(dim=1).clamp(min=1e-9)).squeeze().numpy()

    # Normalize to unit length so cosine similarity = just a dot product
    sentence_vector = sentence_vector / np.linalg.norm(sentence_vector)

    return sentence_vector, tokens


# Show tokenization examples
print("\n" + "="*60)
print("TOKENIZATION EXAMPLES")
print("="*60)

for text in ["Bin played amazing in the finals", "BLG dominated G2", "worst team in the tournament"]:
    vec, tokens = embed_text(text)
    print(f"\n\"{text}\"")
    print(f"  Tokens: {tokens}")
    print(f"  IDs: {tokenizer.encode(text)}")
    print(f"  Vector (first 5): {vec[:5].round(4)}")


# Embed all comments
print(f"\n\nEncoding {len(df)} comments...")
all_embeddings = []
for i, body in enumerate(df['body'].tolist()):
    vec, _ = embed_text(str(body))
    all_embeddings.append(vec)
    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{len(df)}")

all_embeddings = np.array(all_embeddings)
print(f"Done! Shape: {all_embeddings.shape}")


# Store in ChromaDB — passing our own vectors instead of letting ChromaDB encode
client = chromadb.Client()
collection = client.create_collection("comments")

collection.add(
    documents=df['body'].tolist(),
    embeddings=all_embeddings.tolist(),
    metadatas=[{"match": m, "score": int(s)} for m, s in zip(df['match'], df['score'])],
    ids=[str(i) for i in range(len(df))]
)

print(f"\n✅ Ready — {len(df)} embeddings in ChromaDB")
print("="*60)

while True:
    query = input("\nSearch: ")
    if query.lower() == 'quit':
        break

    # Embed query ourselves, then pass the vector to ChromaDB (not query_texts)
    query_vector, query_tokens = embed_text(query)
    print(f"  Tokens: {query_tokens}")
    print(f"  Vector (first 5): {query_vector[:5].round(4)}")

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=5
    )

    for i in range(len(results['documents'][0])):
        print(f"\n--- Result {i+1} (Distance: {results['distances'][0][i]:.4f}) ---")
        print(f"Match: {results['metadatas'][0][i]['match']}")
        print(f"Body: {results['documents'][0][i][:200]}")
