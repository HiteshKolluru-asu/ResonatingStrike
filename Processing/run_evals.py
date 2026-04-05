import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import multi_collection_groq as mcg

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

EVAL_PAIRS = [
    # Basic Facts
    {"q": "Who won the First Stand 2026 Finals?", "expected": "BLG (Bilibili Gaming)"},
    {"q": "What role does Bin play?", "expected": "Top"},
    {"q": "Who was the MVP of the finals?", "expected": "Bin"},
    {"q": "What was the score of the finals?", "expected": "BLG beat G2 3-1"},
    {"q": "Which team was SkewMond on?", "expected": "G2 Esports"},
    {"q": "Who coached BLG?", "expected": "Daeny"},
    {"q": "What was the venue for the tournament?", "expected": "São Paulo, Brazil"},
    {"q": "Who did JDG lose to in the Semis?", "expected": "BLG (Bilibili Gaming)"},
    {"q": "Who got swept in the semi-finals?", "expected": "Gen.G got swept by G2"},
    
    # Meta / Stats
    {"q": "How many games were played on Yunara?", "expected": "30 games. She was played as a support."},
    {"q": "Is Yunara a support or an ADC?", "expected": "Support"},
    {"q": "What role was Zaahen played in?", "expected": "Top and Jungle"},
    
    # Advanced / Nuance
    {"q": "Who played Zaahen the best?", "expected": "Bin (or it's mentioned prominently in context). Mention Zaahen's winrate or presence."},
    {"q": "What did the community think of Caps in the finals?", "expected": "He completely dominated/gapped knight."},
    {"q": "What nationality is BrokenBlade?", "expected": "German/Turkish"},
    {"q": "Name one champion that ON played.", "expected": "Anivia, Bard, or Rumble"},
    {"q": "What is Chovy's signature champion?", "expected": "Azir, Corki, Ahri"},
    {"q": "What happened to GenG in Group A?", "expected": "They progressed to semis but lost the decider to G2"},
    
    # Tricky / Fallback expected
    {"q": "Did Faker play in this tournament?", "expected": "No / I don't have enough reliable data."},
    {"q": "Who won the 2028 World Championship?", "expected": "Cannot be answered / Fallback."}
]

def grade_response(question, expected, actual):
    """Use generating LLM to grade itself stringently."""
    prompt = (
        "You are an evaluator grading a Q&A bot.\n"
        f"Question: {question}\n"
        f"Expected Concept: {expected}\n"
        f"Actual Answer: {actual}\n\n"
        "Did the Actual Answer accurately convey the Expected Concept? Ignore formatting/tone, focus only on facts. "
        "Does the actual answer contain hallucinations? "
        "Reply with EXACTLY the word 'PASS' or 'FAIL'."
    )
    
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
        )
        grading = res.json()["choices"][0]["message"]["content"].strip().upper()
        return "PASS" if "PASS" in grading else "FAIL"
    except Exception as e:
        return "ERROR"

def run_evals():
    print("⏳ Initializing Eval Pipeline...")
    
    # 1. Init everything explicitly (skip the while loop in main)
    df, comment_chunks = mcg.load_comment_chunks()
    player_chunks = mcg.load_player_chunks()
    champion_chunks = mcg.load_champion_chunks()
    tournament_chunks = mcg.load_tournament_chunks()

    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    collections = {
        "comments": client.get_collection("comments"),
        "players": client.get_collection("players"),
        "champions": client.get_collection("champions"),
        "tournament": client.get_collection("tournament"),
    }
    
    tokenized_comments = df["body"].str.lower().str.split()
    bm25 = BM25Okapi(tokenized_comments)

    print(f"🚀 Pipeline loaded. Running {len(EVAL_PAIRS)} tests...\n")
    
    passed = 0
    
    for i, pair in enumerate(EVAL_PAIRS):
        question = pair["q"]
        expected = pair["expected"]
        
        # Retrieval
        results = mcg.multi_collection_search(question, collections, df, bm25)
        
        # Check Fallback logic
        max_hybrid_score = max([c["score"] for c in results["comments"]]) if results["comments"] else 0
        if max_hybrid_score < 0.1 and not results["tournament"] and not results["champions"] and not results["players"]:
            answer = "I don't have enough reliable data from the tournament or comments to answer that confidently."
        else:
            # Build prompt & Ask
            system_prompt = mcg.build_prompt(question, results)
            res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": f"Based on the following context, please answer the question:\n{system_prompt}\n\nQuestion: {question}"}],
                    "temperature": 0.3,
                }
            )
            answer = res.json()["choices"][0]["message"]["content"]
            
        # Evaluation
        grade = grade_response(question, expected, answer)
        if grade == "PASS":
            passed += 1
            print(f"✅ Q{i+1}: {question}")
        else:
            print(f"❌ Q{i+1}: {question}")
            print(f"   Expected: {expected}")
            print(f"   Actual:   {answer[:100]}...\n")
            
        time.sleep(1) # Prevent heavy rate limiting
        
    print(f"\n📊 FINAL SCORE: {passed}/{len(EVAL_PAIRS)} ({(passed/len(EVAL_PAIRS))*100:.1f}%)")

if __name__ == "__main__":
    run_evals()
