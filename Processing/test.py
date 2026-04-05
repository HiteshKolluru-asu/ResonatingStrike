import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
import os



df = pd.read_csv("../GatheringData/cleaned_resonating_strike_data.csv")

# Ensure the columns you use ('body' and 'match') exist
if 'body' not in df.columns:
    print("ERROR: Your file must have a column named 'body'.")
    exit()

# ==========================================
# 2. SETUP THE AI MODEL (NO LOGIN REQUIRED)
# ==========================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "opensearch-project/opensearch-neural-sparse-encoding-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
model.eval()

def get_sparse_vector(text):
    """Converts a string into a neural sparse vector."""
    inputs = tokenizer(
        str(text), 
        return_tensors='pt', 
        truncation=True, 
        padding=True, 
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Neural Sparse Logic: weight = log(1 + relu(logits))
    logits = outputs.logits
    weights = torch.log(1 + torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1)
    sparse_vector = torch.max(weights, dim=1).values
    
    return sparse_vector.squeeze().cpu().numpy()

# ==========================================
# 3. ENCODE THE DATA
# ==========================================
print(f"Encoding {len(df)} rows... (This stays local and private)")
vectors = []
for i, body in enumerate(df['body'].tolist()):
    vectors.append(get_sparse_vector(body))
    if (i + 1) % 100 == 0:
        print(f"Progress: {i + 1}/{len(df)}")

doc_vectors = np.array(vectors)

# ==========================================
# 4. SEARCH LOOP
# ==========================================
print("\n--- Neural Search Ready (Type 'quit' to stop) ---")

while True:
    query_text = input("\nSearch: ")
    if query_text.lower() == 'quit':
        break
    
    query_vec = get_sparse_vector(query_text)
    scores = np.dot(doc_vectors, query_vec)
    
    # Get top 5 indices
    top_n_indices = np.argsort(scores)[::-1][:5]
    
    for rank, idx in enumerate(top_n_indices, 1):
        score = scores[idx]
        if score > 0:
            print(f"\n--- Result {rank} (Score: {score:.2f}) ---")
            
            # Print 'match' column if it exists, otherwise just the index
            match_val = df.iloc[idx]['match'] if 'match' in df.columns else f"Row {idx}"
            print(f"Match: {match_val}")
            
            body_preview = str(df.iloc[idx]['body'])[:200].replace('\n', ' ')
            print(f"Body: {body_preview}...")
        else:
            if rank == 1: print("No relevant matches found.")
            break