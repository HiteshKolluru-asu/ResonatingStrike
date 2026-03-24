import requests
import pandas as pd
import re
import time

# ALL 13 SERIES OF FIRST STAND 2026
FACEOFFS = {
    "Finals: BLG vs G2": "1s0q6q9",
    "Semi 1: G2 vs GEN": "1rzubus",
    "Semi 2: BLG vs JDG": "1rzyiwu",
    "GrpA Decider: G2 vs BFX": "1ryzd7v",
    "GrpB Decider: JDG vs LYON": "1rz5lbx",
    "GrpB Elim: JDG vs LOUD": "1ry6wme",
    "GrpB Winners: GEN vs LYON": "1ry3fjn",
    "GrpA Winners: BLG vs G2": "1rx6wxk",
    "GrpA Elim: BFX vs TSW": "1s14v5v",
    "GrpB Open: LYON vs LOUD": "1rwgzni",
    "GrpB Open: GEN vs JDG": "1rw9cao",
    "GrpA Open: BLG vs BFX": "1rv7xej", # Shared Day 1 Thread
    "GrpA Open: G2 vs TSW": "1rv7xej"
}

HEADERS = {'User-Agent': 'Mozilla/5.0 (ResonatingStrike_FinalScout)'}
all_comments = []

def scrape_thread(thread_id, match_label, is_main=True):
    url = f"https://www.reddit.com/r/leagueoflegends/comments/{thread_id}/.json?limit=100"
    print(f"📡 Scouting: {match_label}...")
    
    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200: return [], []
    
    json_data = res.json()
    raw_comments = json_data[1]['data']['children']
    
    game_links = []
    if is_main:
        # Pinned comments often have Game 1, Game 2 links
        for c in raw_comments[:3]:
            body = c['data'].get('body', '')
            found = re.findall(r'comments/([a-z0-9]+)/', body)
            if found:
                game_links = list(dict.fromkeys(found))
                break

    data = []
    for c in raw_comments:
        if c['kind'] == 't1':
            d = c['data']
            data.append({
                "match": match_label,
                "author": d.get('author'),
                "body": d.get('body'),
                "score": d.get('score')
            })
    return data, game_links

# GATHER EVERYTHING
for name, m_id in FACEOFFS.items():
    main_data, game_ids = scrape_thread(m_id, f"{name} (Series)")
    all_comments.extend(main_data)
    
    for i, g_id in enumerate(game_ids, 1):
        time.sleep(2) # Avoid the Rate-Limit Elder Dragon
        game_data, _ = scrape_thread(g_id, f"{name} - Game {i}", is_main=False)
        all_comments.extend(game_data)

# Save to the master file
df = pd.DataFrame(all_comments)
df = df[df['body'].str.len() > 15].drop_duplicates(subset=['body'])
df.to_csv("resonating_strike_data.csv", index=False)

print(f"\n🏆 FULL TOURNAMENT CAPTURED: {len(df)} comments.")