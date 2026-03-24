# Resonating Strike

A RAG (Retrieval-Augmented Generation) pipeline that answers questions about the **League of Legends First Stand 2026** tournament using real community reactions scraped from Reddit.

Ask it "How did Bin perform in the finals?" and it retrieves the most relevant comments from 1,500+ fan reactions, then uses a local LLM to generate an answer grounded in what the community actually said.

## How It Works

```
User Question
     |
     v
+------------------+
| Hybrid Search    |
| BM25 (keywords)  |  --> Top 5 most relevant comments
| + Vectors (meaning)|
+------------------+
     |
     v
+------------------+
| LLM (llama3)     |  --> Natural language answer
| + Tournament context|     based on retrieved comments
+------------------+
```

**BM25** finds exact keyword matches (searching "Bin" finds comments mentioning "Bin").
**Semantic search** finds meaning matches (searching "dominated" finds comments saying "absolute top gap").
**Hybrid** combines both, normalized to 0-1 scale, weighted equally.

The LLM receives the retrieved comments plus tournament metadata (rosters, bracket results, champion stats) and generates a concise answer.

## Project Structure

```
ResonatingStrike/
├── GatheringData/
│   ├── getData.py                          # Scrapes Reddit comments from all 13 series
│   ├── cleanData.py                        # Removes duplicates, deleted content, link dumps
│   ├── resonating_strike_data.csv          # Raw data (1,565 comments)
│   └── cleaned_resonating_strike_data.csv  # Cleaned data (~1,564 comments)
├── Processing/
│   ├── bm25_Search.py                      # Standalone BM25 keyword search
│   ├── semantic_search.py                  # Standalone vector/embedding search
│   └── hybrid_search.py                    # Full pipeline: hybrid search + LLM generation
├── Project.md                              # Internal project notes
└── README.md
```

## Tech Stack

- **Python 3** + pandas, numpy
- **rank-bm25** - BM25Okapi keyword search
- **sentence-transformers** - `all-MiniLM-L6-v2` for embeddings (384-dim vectors)
- **ChromaDB** - in-memory vector store
- **Ollama** + llama3 (8B) - local LLM for answer generation

## Setup

```bash
# Clone and enter project
cd ResonatingStrike

# Set up virtual environment
python3 -m venv GatheringData/venv
source GatheringData/venv/bin/activate

# Install dependencies
pip install pandas rank-bm25 sentence-transformers chromadb requests numpy

# Install Ollama (https://ollama.com) then pull the model
ollama pull llama3
```

## Usage

```bash
# Make sure Ollama is running
ollama serve

# Run the full RAG pipeline
source GatheringData/venv/bin/activate
cd Processing
python3 hybrid_search.py
```

Then ask questions like:
- `Who was the best player in the tournament?`
- `How did G2 beat GenG?`
- `What did people think about Bin's Jax?`
- `best top laner`

Type `quit` to exit.

## Data

1,564 cleaned comments from Reddit's r/leagueoflegends covering all 13 series of First Stand 2026:
- Group stage (8 matches)
- Semifinals (2 matches)
- Finals (BLG vs G2)

Each comment includes: match name, comment body, and Reddit score (upvotes).

## Limitations

- **Small dataset** - 1,564 comments limits retrieval quality. Niche queries may return weak results. A larger dataset (100K+ comments) would significantly improve answer quality.
- **Dynamic context** - Retrieves up to 20 relevant comments per query (filtered by a 0.1 relevance threshold), but niche topics may still have few matches.
- **General embedding model** - `all-MiniLM-L6-v2` doesn't understand League-specific slang (e.g., "purple monster" for Baron Nashor).
- **No persistence** - ChromaDB runs in-memory, so embeddings are regenerated each run.

Despite these limitations, llama3 8B does a surprisingly good job of synthesizing answers from minimal context — a testament to how well even small local models perform when given the right retrieval pipeline.
