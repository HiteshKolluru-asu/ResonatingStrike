# Resonating Strike

A Multi-Collection RAG (Retrieval-Augmented Generation) pipeline that answers questions about the **League of Legends First Stand 2026** tournament by blending raw, verified tournament facts with real community reactions scraped from Reddit.

Ask it "Who played Zaahen the best?" or "How did Bin perform in the finals?" and it retrieves the most relevant knowledge chunks—player profiles, champion stats, match results, and fan reactions—then generates a natural, esports-analyst style answer using local or cloud-based LLMs.

## How It Works

```
User Question
     |
     v
+-----------------------------+
| Multi-Collection Retrieval  |
| 1. Comments (Hybrid Search) |
| 2. Players (Semantic)       |
| 3. Champions (Semantic)     |
| 4. Tournament (Semantic)    |
+-----------------------------+
     |
     v
+-----------------------------+
| LLM (Groq / Ollama)         |  --> Natural language analyst response
| + Blended Context           |
+-----------------------------+
```

The pipeline searches four distinct ChromaDB collections simultaneously:
- **Comments (Hybrid):** Uses BM25 (keyword matching) + Semantic Search (meaning) on 1,500+ Reddit fan reactions, normalized and combined.
- **Players:** Profiles of 40 pro players and 8 coaches for deep roster insights.
- **Champions:** Pick/ban stats, win rates, and meta narratives for 61 champions using Riot's Data Dragon and tournament data.
- **Tournament:** Bracket results, match summaries, MVP awards, and team data.

The retrieved context is compiled into a comprehensive prompt and sent to an LLM, which adopts an expert yet conversational esports analyst persona.

## Project Structure

```
ResonatingStrike/
├── GatheringData/
│   ├── getData.py                          # Scrapes Reddit comments from all 13 series
│   ├── cleanData.py                        # Removes duplicates, deleted content, link dumps
│   └── cleaned_resonating_strike_data.csv  # Cleaned data (~1,564 comments)
├── Knowledge/
│   ├── champions.json                      # 61 champion profiles & tournament stats
│   ├── fetch_champions.py                  # Generates champions.json using Data Dragon
│   ├── players.json                        # 40 player & 8 coach profiles
│   └── tournament.json                     # Bracket results, match summaries, teams
├── Processing/
│   ├── multi_collection_groq.py            # Primary script: Cloud LLM via Groq API (llama-3.3-70b-versatile)
│   ├── multi_collection_search.py          # Local alternative: Local LLM via Ollama (gemma2)
│   ├── hybrid_search.py                    # Legacy: Standalone hybrid search
│   └── semantic_search.py                  # Legacy: Standalone vector/embedding search
├── chroma_db/                              # Persistent ChromaDB storage (auto-generated)
└── README.md
```

## Tech Stack

- **Python 3** + `pandas`, `numpy`, `python-dotenv`
- **Embedding Model**: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (tuned specifically for semantic QA)
- **Vector Store**: **ChromaDB** with persistent local storage
- **Lexical Search**: `rank-bm25` (BM25Okapi)
- **LLM APIs**:
  - Cloud: **Groq API** (`llama-3.3-70b-versatile`) for blazing-fast inference without GPU overhead.
  - Local: **Ollama** (`gemma2`) for offline privacy and local execution.

## Setup

1. **Clone and enter project:**
   ```bash
   git clone <repo-url>
   cd ResonatingStrike
   ```

2. **Set up virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install pandas rank-bm25 sentence-transformers chromadb requests numpy python-dotenv
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_api_key_here
   ```

## Usage

For the best, fastest experience, use the Groq cloud pipeline:

```bash
source venv/bin/activate
cd Processing
python3 multi_collection_groq.py
```

To run it entirely locally (requires Ollama and the `gemma2` model installed):

```bash
source venv/bin/activate
cd Processing
python3 multi_collection_search.py
```

### Features to Try
- **Ask a complex question:** "Who was the best top laner based on community sentiment vs stats?" or "What role was Zaahen played in?"
- **Debug retrieval mode:** Type `debug` and hit enter. The next query you ask will print out exactly which chunks from which databases were retrieved and their similarity scores.

## Recent Upgrades

- **Persistent Vector Storage:** Collections are now saved locally (`./chroma_db`). The heavy lifting of embedding 1,700+ document chunks is only done once, reducing pipeline startup time to less than 1 second.
- **Upgraded Encoder:** Transitioned from `all-MiniLM-L6-v2` to `multi-qa-MiniLM-L6-cos-v1` for superior performance in question-answering context retrieval.
- **Structured Knowledge Graph:** Split the single text blob into four distinct, highly detailed databases.
- **Esports Analyst Persona:** Transformed the LLM's system prompt from standard robotic responses to an authentic, conversational flow that smoothly weaves factual stats with the underlying Reddit context.
