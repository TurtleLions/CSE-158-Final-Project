import pandas as pd
import ast
import random
import numpy as np
import argparse
import sys
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Parse command line arguments for sweep
parser = argparse.ArgumentParser(description='Run model with variable feature parameters')
parser.add_argument('--top_n_tags', type=int, default=25, help='Number of top game tags to track for affinity')
parser.add_argument('--top_n_words', type=int, default=0, help='Number of top review words to track for affinity (0 to disable)')
args = parser.parse_args()

TOP_N_TAGS_COUNT = args.top_n_tags
TOP_N_WORDS_COUNT = args.top_n_words

print(f"Starting Run: Tags={TOP_N_TAGS_COUNT}, Words={TOP_N_WORDS_COUNT}")

# Load data
print("Loading user/item interactions")
user_data = []
with open('australian_users_items.json.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        user_data.append(ast.literal_eval(line))

print("Loading game metadata")
game_data = []
with open('steam_new.json.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        try:
            game_data.append(ast.literal_eval(line))
        except:
            continue

# Load reviews for bag of words
user_reviews = {}
if TOP_N_WORDS_COUNT > 0:
    print("Loading user reviews")
    with open('australian_user_reviews.json.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            review_node = ast.literal_eval(line)
            user_id = str(review_node['user_id'])
            # Aggregate all review text for this user
            full_text = " ".join([r.get('review', '') for r in review_node.get('reviews', [])])
            user_reviews[user_id] = full_text

# Process game metadata
games_dict = {}
all_tags = []

for game in game_data:
    if 'id' in game:
        gid = str(game['id'])
    else:
        continue
    
    price = game.get('price', 0)
    if isinstance(price, str):
        if 'free' in price.lower():
            price = 0
        else:
            try:
                price = float(price)
            except:
                price = 0
                
    tags = game.get('tags', [])
    genres = game.get('genres', [])
    
    # Combine and normalize tags (lowercase for matching with reviews)
    combined_tags = list(set(tags + genres))
    normalized_tags = [t.lower() for t in combined_tags]
    all_tags.extend(combined_tags)
    
    games_dict[gid] = {
        'price': price,
        'tags': combined_tags,
        'tags_lower': set(normalized_tags)
    }

# Tag profiles
top_tags_counter = Counter(all_tags)
TOP_TAGS_LIST = [tag for tag, count in top_tags_counter.most_common(TOP_N_TAGS_COUNT)]
print(f"Top {TOP_N_TAGS_COUNT} tags: {TOP_TAGS_LIST[:5]}...")

interactions = []
user_tag_profiles = {}

for user in user_data:
    user_id = str(user['user_id'])
    items_count = user['items_count']
    
    user_games_tags = []
    
    for item in user['items']:
        gid = str(item['item_id'])
        interactions.append({
            'user_id': user_id,
            'item_id': gid,
            'playtime_forever': item['playtime_forever'],
            'user_items_count': items_count
        })
        
        if gid in games_dict:
            user_games_tags.extend(games_dict[gid]['tags'])

    if user_games_tags:
        total = len(user_games_tags)
        counts = Counter(user_games_tags)
        user_tag_profiles[user_id] = {t: counts[t]/total for t in TOP_TAGS_LIST if t in counts}
    else:
        user_tag_profiles[user_id] = {}

# Review word profiles and bag of words
user_word_profiles = {}
if TOP_N_WORDS_COUNT > 0:
    print("Building Bag of Words profiles")
    all_words = []
    
    # Tokenizer helper
    def get_tokens(text):
        # Simple regex to get words, ignore short ones like "a"
        return [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text)]

    # First pass through dataset to collect global word counts to find top N
    global_word_counter = Counter()
    for uid, text in user_reviews.items():
        tokens = get_tokens(text)
        global_word_counter.update(tokens)
    
    TOP_WORDS_LIST = [w for w, c in global_word_counter.most_common(TOP_N_WORDS_COUNT)]
    TOP_WORDS_SET = set(TOP_WORDS_LIST)
    print(f"Top {TOP_N_WORDS_COUNT} words: {TOP_WORDS_LIST[:5]}...")

    # Second pass to build user specific profiles
    for uid, text in user_reviews.items():
        tokens = get_tokens(text)
        # Filter only top words
        relevant_tokens = [t for t in tokens if t in TOP_WORDS_SET]
        if relevant_tokens:
            total = len(relevant_tokens)
            counts = Counter(relevant_tokens)
            # Normalize
            user_word_profiles[uid] = {w: counts[w]/total for w in counts}
        else:
            user_word_profiles[uid] = {}

# make df
df = pd.DataFrame(interactions)
df = df[df['playtime_forever'] > 0] 

item_popularity = df.groupby('item_id')['user_id'].nunique().reset_index()
item_popularity.columns = ['item_id', 'item_popularity']
df = df.merge(item_popularity, on='item_id', how='left')

# Negative sampling
played_pairs = set(zip(df['user_id'], df['item_id']))
all_item_ids = list(df['item_id'].unique())
users = df['user_id'].values
n_samples = len(users)

negative_items = np.empty(n_samples, dtype=object)
filled_mask = np.zeros(n_samples, dtype=bool)
max_retries = 100 
retry_count = 0

while not filled_mask.all() and retry_count < max_retries:
    missing_count = (~filled_mask).sum()
    candidates = np.random.choice(all_item_ids, size=missing_count)
    current_users = users[~filled_mask]
    valid_mask = np.array([ (u, i) not in played_pairs for u, i in zip(current_users, candidates) ])
    if valid_mask.any():
        update_indices = np.where(~filled_mask)[0][valid_mask]
        negative_items[update_indices] = candidates[valid_mask]
        filled_mask[update_indices] = True
    retry_count += 1

df_negative = pd.DataFrame({
    'user_id': users,
    'item_id': negative_items,
    'target': 0
})
df_negative['user_items_count'] = df['user_items_count'].values 
df_negative = df_negative.merge(item_popularity, on='item_id', how='left')

df_positive = df[['user_id', 'item_id', 'user_items_count', 'item_popularity']].copy()
df_positive['target'] = 1

df_model = pd.concat([df_positive, df_negative], ignore_index=True)
df_model = df_model.dropna()

# Calculate features
def calculate_tag_affinity(row):
    uid = row['user_id']
    gid = row['item_id']
    if uid not in user_tag_profiles or gid not in games_dict: return 0
    profile = user_tag_profiles[uid]
    game_tags = games_dict[gid]['tags']
    score = 0
    for tag in game_tags:
        if tag in profile:
            score += profile[tag]
    return score

def calculate_review_affinity(row):
    # Matches user review words -> game tags ("action" to "Action")
    if TOP_N_WORDS_COUNT == 0: return 0
    uid = row['user_id']
    gid = row['item_id']
    
    # Check for profiles
    if uid not in user_word_profiles: return 0
    if gid not in games_dict: return 0
    
    profile = user_word_profiles[uid] # Dict of {word: weight}
    game_tags_lower = games_dict[gid]['tags_lower'] # Set of lowercase tags
    
    score = 0
    for word, weight in profile.items():
        if word in game_tags_lower:
            score += weight
    return score

print("Calculating affinity scores")
df_model['affinity_score'] = df_model.apply(calculate_tag_affinity, axis=1)

if TOP_N_WORDS_COUNT > 0:
    print("Calculating review affinity scores")
    df_model['review_affinity_score'] = df_model.apply(calculate_review_affinity, axis=1)

def get_price(gid):
    return games_dict.get(gid, {}).get('price', 0)
df_model['price'] = df_model['item_id'].apply(get_price)

# Training
features = ['user_items_count', 'item_popularity', 'price', 'affinity_score']
if TOP_N_WORDS_COUNT > 0:
    features.append('review_affinity_score')

X = df_model[features]
y = df_model['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred_probability = model.predict_proba(X_test_scaled)[:, 1]

auc_score = roc_auc_score(y_test, y_pred_probability)

# Output CSV format for the bash script to capture
# RESULT: is what is searched for by grid_sweep.sh
print(f"RESULT: Tags={TOP_N_TAGS_COUNT}, Words={TOP_N_WORDS_COUNT}, AUC={auc_score:.4f}")

# Plot code (not needed for now)
# plt.figure(figsize=(10, 5))
# sns.barplot(x=features, y=np.abs(model.coef_[0]))
# plt.title(f'Feature Importance (Tags={TOP_N_TAGS_COUNT}, Words={TOP_N_WORDS_COUNT})')
# plt.savefig(f'feat_imp_t{TOP_N_TAGS_COUNT}_w{TOP_N_WORDS_COUNT}.png')