"""
Fetch champion data from Riot's Data Dragon API and combine with 
tournament-specific pick/ban statistics for First Stand 2026.

Data Dragon API is free and requires no API key.
Run this script from the Knowledge/ directory.
"""

import urllib.request
import json
import os

# --- Configuration ---
DDRAGON_VERSION = "16.7.1"  # Latest Data Dragon version
DDRAGON_URL = f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VERSION}/data/en_US/champion.json"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "champions.json")

# Champions that appeared (picked OR banned) at First Stand 2026
# Mapped as: DataDragon key -> tournament display name
# This list is curated from tournament pick/ban data
TOURNAMENT_CHAMPIONS = {
    # --- HIGH PRIORITY (>50% presence) ---
    "Orianna": "Orianna",
    "Ryze": "Ryze",
    "XinZhao": "Xin Zhao",
    "Yunara": "Yunara",
    "Rumble": "Rumble",
    "Varus": "Varus",
    "Karma": "Karma",

    # --- MEDIUM PRIORITY (20-50% presence) ---
    "Vi": "Vi",
    "Ahri": "Ahri",
    "Aurora": "Aurora",

    # --- PICKED / NOTABLE ---
    # Top lane
    "Jax": "Jax",
    "Camille": "Camille",
    "Renekton": "Renekton",
    "Aatrox": "Aatrox",
    "Gnar": "Gnar",
    "Shen": "Shen",
    "Kled": "Kled",
    "Jayce": "Jayce",
    "Kennen": "Kennen",
    "Ornn": "Ornn",
    "Garen": "Garen",
    "Zaahen": "Zaahen",

    # Jungle
    "Viego": "Viego",
    "LeeSin": "Lee Sin",
    "Nidalee": "Nidalee",
    "Graves": "Graves",
    "Sejuani": "Sejuani",

    # Mid lane
    "Syndra": "Syndra",
    "Zoe": "Zoe",
    "Anivia": "Anivia",
    "Vex": "Vex",
    "Azir": "Azir",
    "Corki": "Corki",
    "Taliyah": "Taliyah",
    "Yone": "Yone",
    "Yasuo": "Yasuo",

    # Bot lane
    "Jhin": "Jhin",
    "Aphelios": "Aphelios",
    "Ashe": "Ashe",
    "Jinx": "Jinx",
    "Xayah": "Xayah",
    "Tristana": "Tristana",
    "Kalista": "Kalista",
    "Draven": "Draven",
    "Zeri": "Zeri",
    "MissFortune": "Miss Fortune",
    "Kaisa": "Kai'Sa",
    "Lucian": "Lucian",

    # Support
    "Nautilus": "Nautilus",
    "Bard": "Bard",
    "Neeko": "Neeko",
    "Seraphine": "Seraphine",
    "Lulu": "Lulu",
    "Nami": "Nami",
    "Rakan": "Rakan",
    "Thresh": "Thresh",
    "Alistar": "Alistar",
    "Braum": "Braum",
    "Rell": "Rell",
    "Poppy": "Poppy",
    "Maokai": "Maokai",
}

# Tournament-specific statistics (from gol.gg / community data)
# pb_pct = pick/ban %, games = games played, bans = times banned
# roles = list of roles played at the tournament (first = primary)
TOURNAMENT_STATS = {
    # Highest presence
    "Orianna":   {"pb_pct": 82.2, "bans": 33, "games": 4,  "wins": 2, "roles": ["Mid"]},
    "Ryze":      {"pb_pct": 82.2, "bans": 34, "games": 3,  "wins": 1, "roles": ["Mid", "Top"]},
    "Xin Zhao":  {"pb_pct": 80.0, "bans": 6,  "games": 30, "wins": 16, "roles": ["Jungle"]},
    "Yunara":     {"pb_pct": 77.8, "bans": 5,  "games": 30, "wins": 15, "roles": ["Bot"]},
    "Rumble":     {"pb_pct": 68.9, "bans": 23, "games": 8,  "wins": 4, "roles": ["Top", "Support"]},
    "Varus":     {"pb_pct": 66.7, "bans": 22, "games": 8,  "wins": 3, "roles": ["Bot"]},
    "Karma":     {"pb_pct": 66.7, "bans": 25, "games": 5,  "wins": 3, "roles": ["Support"]},

    # High presence
    "Vi":        {"pb_pct": 46.7, "bans": 9,  "games": 12, "wins": 6, "roles": ["Jungle"]},
    "Ahri":      {"pb_pct": 42.2, "bans": 4,  "games": 15, "wins": 8, "roles": ["Mid"]},
    "Aurora":    {"pb_pct": 28.9, "bans": 3,  "games": 10, "wins": 5, "roles": ["Mid", "Top"]},

    # Significant picks
    "Jax":       {"pb_pct": 24.4, "bans": 5,  "games": 6,  "wins": 4, "roles": ["Top"]},
    "Camille":   {"pb_pct": 20.0, "bans": 3,  "games": 6,  "wins": 3, "roles": ["Top"]},
    "Renekton":  {"pb_pct": 17.8, "bans": 2,  "games": 6,  "wins": 4, "roles": ["Top"]},
    "Nautilus":  {"pb_pct": 35.6, "bans": 4,  "games": 12, "wins": 6, "roles": ["Support"]},
    "Bard":      {"pb_pct": 26.7, "bans": 6,  "games": 6,  "wins": 3, "roles": ["Support"]},
    "Syndra":    {"pb_pct": 22.2, "bans": 4,  "games": 6,  "wins": 3, "roles": ["Mid"]},
    "Aphelios":  {"pb_pct": 26.7, "bans": 2,  "games": 10, "wins": 5, "roles": ["Bot"]},
    "Jhin":      {"pb_pct": 20.0, "bans": 1,  "games": 8,  "wins": 3, "roles": ["Bot"]},
    "Shen":      {"pb_pct": 15.6, "bans": 1,  "games": 6,  "wins": 2, "roles": ["Top"]},
    "Viego":     {"pb_pct": 17.8, "bans": 2,  "games": 6,  "wins": 3, "roles": ["Jungle"]},
    "Lee Sin":   {"pb_pct": 15.6, "bans": 1,  "games": 6,  "wins": 3, "roles": ["Jungle"]},
    "Zoe":       {"pb_pct": 11.1, "bans": 1,  "games": 4,  "wins": 3, "roles": ["Mid"]},
    "Ashe":      {"pb_pct": 13.3, "bans": 0,  "games": 6,  "wins": 3, "roles": ["Bot"]},
    "Jinx":      {"pb_pct": 15.6, "bans": 1,  "games": 6,  "wins": 3, "roles": ["Bot"]},
    "Rakan":     {"pb_pct": 15.6, "bans": 2,  "games": 5,  "wins": 2, "roles": ["Support"]},
    "Thresh":    {"pb_pct": 11.1, "bans": 0,  "games": 5,  "wins": 2, "roles": ["Support"]},
    "Neeko":     {"pb_pct": 17.8, "bans": 2,  "games": 6,  "wins": 3, "roles": ["Support", "Mid"]},
    "Seraphine": {"pb_pct": 15.6, "bans": 1,  "games": 6,  "wins": 4, "roles": ["Support", "Bot"]},
    "Gnar":      {"pb_pct": 13.3, "bans": 0,  "games": 6,  "wins": 2, "roles": ["Top"]},
    "Aatrox":    {"pb_pct": 11.1, "bans": 1,  "games": 4,  "wins": 2, "roles": ["Top"]},
    "Graves":    {"pb_pct": 11.1, "bans": 0,  "games": 5,  "wins": 2, "roles": ["Jungle", "Top"]},
    "Azir":      {"pb_pct": 13.3, "bans": 2,  "games": 4,  "wins": 2, "roles": ["Mid"]},
    "Lulu":      {"pb_pct": 13.3, "bans": 1,  "games": 5,  "wins": 3, "roles": ["Support"]},
    "Nami":      {"pb_pct": 11.1, "bans": 1,  "games": 4,  "wins": 2, "roles": ["Support"]},
    "Sejuani":   {"pb_pct": 8.9,  "bans": 0,  "games": 4,  "wins": 2, "roles": ["Jungle"]},
    "Xayah":     {"pb_pct": 8.9,  "bans": 0,  "games": 4,  "wins": 2, "roles": ["Bot"]},
    "Kai'Sa":    {"pb_pct": 8.9,  "bans": 0,  "games": 4,  "wins": 2, "roles": ["Bot"]},
    "Corki":     {"pb_pct": 8.9,  "bans": 0,  "games": 4,  "wins": 1, "roles": ["Mid"]},
    "Tristana":  {"pb_pct": 6.7,  "bans": 0,  "games": 3,  "wins": 1, "roles": ["Bot"]},
    "Alistar":   {"pb_pct": 6.7,  "bans": 0,  "games": 3,  "wins": 1, "roles": ["Support"]},
    "Maokai":    {"pb_pct": 6.7,  "bans": 0,  "games": 3,  "wins": 2, "roles": ["Support", "Jungle"]},
    "Poppy":     {"pb_pct": 6.7,  "bans": 1,  "games": 2,  "wins": 1, "roles": ["Jungle", "Support"]},
    "Anivia":    {"pb_pct": 6.7,  "bans": 0,  "games": 3,  "wins": 1, "roles": ["Mid", "Support"]},
    "Zaahen":    {"pb_pct": 15.6, "bans": 2,  "games": 5,  "wins": 3, "roles": ["Top", "Jungle"]},

    # Low presence / Fearless Draft one-offs
    "Vex":       {"pb_pct": 4.4,  "bans": 0,  "games": 2,  "wins": 0, "roles": ["Mid"]},
    "Kled":      {"pb_pct": 4.4,  "bans": 0,  "games": 2,  "wins": 1, "roles": ["Top"]},
    "Braum":     {"pb_pct": 4.4,  "bans": 0,  "games": 2,  "wins": 1, "roles": ["Support"]},
    "Rell":      {"pb_pct": 4.4,  "bans": 0,  "games": 2,  "wins": 1, "roles": ["Support"]},
    "Nidalee":   {"pb_pct": 4.4,  "bans": 0,  "games": 2,  "wins": 1, "roles": ["Jungle"]},
    "Jayce":     {"pb_pct": 4.4,  "bans": 0,  "games": 2,  "wins": 1, "roles": ["Top"]},
    "Kennen":    {"pb_pct": 4.4,  "bans": 0,  "games": 2,  "wins": 1, "roles": ["Top"]},
    "Ornn":      {"pb_pct": 4.4,  "bans": 0,  "games": 2,  "wins": 0, "roles": ["Top"]},
    "Taliyah":   {"pb_pct": 4.4,  "bans": 0,  "games": 2,  "wins": 1, "roles": ["Mid", "Jungle"]},
    "Kalista":   {"pb_pct": 2.2,  "bans": 0,  "games": 1,  "wins": 0, "roles": ["Bot"]},
    "Draven":    {"pb_pct": 2.2,  "bans": 0,  "games": 1,  "wins": 1, "roles": ["Bot"]},
    "Zeri":      {"pb_pct": 2.2,  "bans": 0,  "games": 1,  "wins": 0, "roles": ["Bot"]},
    "Yone":      {"pb_pct": 2.2,  "bans": 0,  "games": 1,  "wins": 0, "roles": ["Mid"]},
    "Yasuo":     {"pb_pct": 2.2,  "bans": 0,  "games": 1,  "wins": 0, "roles": ["Mid"]},
    "Lucian":    {"pb_pct": 2.2,  "bans": 0,  "games": 1,  "wins": 0, "roles": ["Bot", "Mid"]},
    "Garen":     {"pb_pct": 2.2,  "bans": 0,  "games": 1,  "wins": 1, "roles": ["Top"]},
    "Miss Fortune": {"pb_pct": 2.2, "bans": 0, "games": 1, "wins": 0, "roles": ["Bot"]},
}

# Tournament context for notable champions
TOURNAMENT_CONTEXT = {
    "Jax": "Bin's signature pick — he dominated with it in the finals to earn MVP. A must-ban against BLG in later rounds.",
    "Camille": "Part of Bin's champion ocean. BLG 'overcooked with Camille' in some games per community analysis.",
    "Orianna": "Extremely high ban rate (61%) — teams didn't want to face her in the Fearless Draft meta. When she got through, she was impactful.",
    "Ryze": "Banned out in 34 of 45 games. One of the most feared mid lane picks of the tournament, barely allowed on stage.",
    "Xin Zhao": "Workhorse jungle pick — appeared in the vast majority of games. Reliable engage and early game power.",
    "Yunara": "Bot lane marksman (ADC) who appeared in every single series of the tournament. Highest presence ADC at 77.8% — teams either first-picked or banned her. Defined the bot lane meta at First Stand 2026.",
    "Rumble": "High priority ban, used primarily in top lane. When picked, provided strong teamfight ultimates.",
    "Varus": "Contested ADC pick but notable for Hans Sama's struggles — 'couldn't hit ultimates, wasn't dealing damage.'",
    "Karma": "Top priority support with high win rate. Valued for shields, speed-ups, and poke. Defined the enchanter side of the support meta.",
    "Vi": "Reliable jungle option, picked consistently across all teams. Strong engage and pick potential.",
    "Ahri": "Flexible mid lane pick with charm-based pick potential. Consistently useful across different team compositions.",
    "Aurora": "Newer champion that saw moderate play. Mid lane flex pick with unique mobility.",
    "Zoe": "Caps' playground — used to 'rub it in' against BFX in the Group A decider. Sleepy trouble bubble highlight plays.",
    "Syndra": "knight's standout champion — 'Giving knight a good Syndra game' was the community reaction.",
    "Shen": "Memed for doing '0 damage' in some games. Utility-focused top lane pick.",
    "Jhin": "'Late game Jhin strikes again' — criticized for being a scaling ADC in a meta that rewarded early aggression.",
    "Vex": "'Fucks it up playing Vex smh' — underperformed at the tournament. Anti-mobility mage that didn't find its niche.",
    "Kled": "'Someone watched a how-to-play Kled video during champ select' — a meme about an unexpected top lane pick.",
    "Renekton": "Strong top lane bruiser pick. Teams used Renekton to secure early lane advantages and snowball games.",
    "Bard": "Prominent support pick at the tournament. 'Bard and Rumble are out now' — noted as falling out of meta after the event.",
    "Aphelios": "Complex ADC that appeared primarily in bot lane matchups. 'Shooting weird bubble guns' per community.",
    "Nautilus": "Core engage support. Consistently picked across all teams for hook threats and crowd control.",
    "Viego": "Canyon's signature champion in previous years, but he didn't perform on it at this tournament.",
    "Anivia": "Mostly played mid lane, but ON (BLG's support) pulled out the support Anivia — 'Support Anivia. Coming to a rift near you soon' was the community reaction.",
    "Zaahen": "Darkin bruiser flexed between top and jungle. In Finals Game 2: 'Zaahen with 5 more minutes would literally solo the game.' Xiaoxu played it top lane; also featured in Group B matches. Part of the Ambessa/Zaahen top lane meta.",
}


def fetch_champion_data():
    """Fetch champion base data from Data Dragon API."""
    print(f"📡 Fetching champion data from Data Dragon v{DDRAGON_VERSION}...")
    with urllib.request.urlopen(DDRAGON_URL) as response:
        data = json.loads(response.read().decode())
    print(f"   Found {len(data['data'])} champions total")
    return data['data']


def build_champions_json(ddragon_data):
    """Build the champions.json with Data Dragon info + tournament stats."""
    champions = {}

    for dd_key, display_name in TOURNAMENT_CHAMPIONS.items():
        champ_data = ddragon_data.get(dd_key)

        if not champ_data:
            print(f"  ⚠️  {dd_key} not found in Data Dragon, using manual entry")
            # Manual entry for champions not in this version of Data Dragon
            entry = {
                "name": display_name,
                "data_dragon_key": dd_key,
                "title": "Unknown",
                "tags": [],
                "blurb": "",
                "image": None,
            }
        else:
            entry = {
                "name": champ_data["name"],
                "data_dragon_key": dd_key,
                "title": champ_data["title"],
                "tags": champ_data["tags"],
                "blurb": champ_data["blurb"],
                "image": f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VERSION}/img/champion/{dd_key}.png",
            }

        # Add tournament stats
        stats = TOURNAMENT_STATS.get(display_name, {})
        roles = stats.get("roles", ["Unknown"])
        entry["tournament"] = {
            "roles": roles,
            "primary_role": roles[0],
            "presence_pct": stats.get("pb_pct", 0),
            "bans": stats.get("bans", 0),
            "games_played": stats.get("games", 0),
            "wins": stats.get("wins", 0),
            "win_rate": round(stats["wins"] / stats["games"] * 100, 1) if stats.get("games", 0) > 0 else None,
        }

        # Add tournament narrative/context
        entry["tournament_context"] = TOURNAMENT_CONTEXT.get(display_name, None)

        champions[display_name] = entry

    return champions


def main():
    # Fetch from API
    ddragon_data = fetch_champion_data()

    # Build combined data
    champions = build_champions_json(ddragon_data)
    print(f"\n🏆 Built data for {len(champions)} tournament champions")

    # Sort by presence (highest first)
    sorted_champs = dict(
        sorted(champions.items(), key=lambda x: x[1]["tournament"]["presence_pct"], reverse=True)
    )

    # Write output
    output = {
        "meta": {
            "tournament": "First Stand 2026",
            "patch": "26.5",
            "data_dragon_version": DDRAGON_VERSION,
            "total_games": 45,
            "total_champions_picked_or_banned": len(sorted_champs),
            "draft_format": "Hard Fearless Draft — no champion repeats within a series",
            "note": "Champion data from Riot Data Dragon API (free). Tournament stats from gol.gg and community data. Some lower-presence stats are estimated."
        },
        "champions": sorted_champs
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved to {OUTPUT_FILE}")

    # Summary
    by_role = {}
    for name, champ in sorted_champs.items():
        role = champ["tournament"]["primary_role"]
        by_role.setdefault(role, []).append(name)

    print(f"\n📊 Champions by role:")
    for role in ["Top", "Jungle", "Mid", "Bot", "Support", "Unknown"]:
        if role in by_role:
            print(f"  {role}: {', '.join(by_role[role])}")


if __name__ == "__main__":
    main()
