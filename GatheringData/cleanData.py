import pandas as pd

df = pd.read_csv("resonating_strike_data.csv")

print(f"Starting with {len(df)} comments")

# Remove duplicate comments
df = df.drop_duplicates(subset=['body'])

# # Remove bot comments 
# commented since we cannot find any bots
# bot_phrases = ['i am a bot', "i'm a bot", 'performed automatically', 'contact the moderators']
# is_bot = df['body'].str.lower().str.contains('|'.join(bot_phrases), na=False)
# print(df[is_bot][['author', 'body', 'score']])
# df = df[~is_bot]
# print(f"After removing bots, we have {len(df)} comments")

# Remove deleted comments
df = df[df['author'] != '[deleted]']
df = df[~df['body'].isin(['[deleted]', '[removed]'])]

# Remove link dumps
df = df[df['body'].str.count('https?://') < 3]
print(f"After removing link dumps, we have {len(df)} comments")

df['body'] = df['body'].str.replace('&gt;', '>', regex=False)
df['body'] = df['body'].str.replace('&#x200B;', '', regex=False)

print(f"After removing markdown artifacts, we have {len(df)} comments")

df.drop(columns=['author'], inplace=True)

# Save the cleaned data
df.to_csv("cleaned_resonating_strike_data.csv", index=False)

print(f"Cleaned data saved to cleaned_resonating_strike_data.csv")