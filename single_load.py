import pandas as pd
import ast
import random
import numpy as np
import sys
import re
import gzip
import os
import gc
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Grid Search Configuration
TAG_COUNTS = [10]
WORD_COUNTS = [0, 50]
C_VALUES = [0.1]

print("Starting Optimized Grid Sweep...")

# Helper function to handle gzip files
def smart_open(file_path, mode='rt', encoding='utf-8'):
    # Try to open the file if the path exists exactly as given
    if os.path.exists(file_path):
        if file_path.endswith('.gz'):
            return gzip.open(file_path, mode=mode, encoding=encoding)
        return open(file_path, mode=mode, encoding=encoding)
    
    # If not found, check if a .gz version exists
    gz_path = file_path + '.gz'
    if os.path.exists(gz_path):
        return gzip.open(gz_path, mode=mode, encoding=encoding)

    # Fallback: try opening original path anyway to raise the standard FileNotFoundError
    return open(file_path, mode=mode, encoding=encoding)

# Load data
print("Loading user/item interactions")
user_data = []
try:
    with smart_open('australian_users_items.json.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            user_data.append(ast.literal_eval(line))
except FileNotFoundError:
    print("Error: australian_users_items.json.gz not found.")
    sys.exit(1)

print("Loading game metadata")
game_data = []
try:
    with smart_open('steam_games.json.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                game_data.append(ast.literal_eval(line))
            except:
                continue
except FileNotFoundError:
    print("Error: steam_games.json.gz not found.")
    sys.exit(1)

# Load reviews for bag of words
user_reviews = {}
# Check if any run requires words
if max(WORD_COUNTS) > 0:
    print("Loading user reviews")
    try:
        with smart_open('australian_user_reviews.json.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                review_node = ast.literal_eval(line)
                user_id = str(review_node['user_id'])
                # Aggregate all review text for this user
                full_text = " ".join([r.get('review', '') for r in review_node.get('reviews', [])])
                user_reviews[user_id] = full_text
    except FileNotFoundError:
        print("Warning: Reviews file not found. Skipping text features.")
        WORD_COUNTS = [0]

# Process game metadata
games_dict = {}
all_tags_global = []

for game in game_data:
    if 'id' in game:
        gid = str(game['id'])
    elif 'app_id' in game:
        gid = str(game['app_id'])
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
    all_tags_global.extend(combined_tags)
    
    games_dict[gid] = {
        'price': price,
        'tags': combined_tags,
        'tags_lower': set(normalized_tags)
    }

# Tag profiles
# Pre-calculate global counts once
global_tag_counter = Counter(all_tags_global)
del game_data # Save RAM
del all_tags_global

interactions = []
user_full_tag_counts = {} # Store full counts to slice later

for user in user_data:
    user_id = str(user['user_id'])
    items_count = user['items_count']
    
    u_tags = []
    
    for item in user['items']:
        gid = str(item['item_id'])
        interactions.append({
            'user_id': user_id,
            'item_id': gid,
            'playtime_forever': item['playtime_forever'],
            'user_items_count': items_count
        })
        
        if gid in games_dict:
            u_tags.extend(games_dict[gid]['tags'])

    if u_tags:
        user_full_tag_counts[user_id] = Counter(u_tags)
    else:
        user_full_tag_counts[user_id] = Counter()

del user_data
gc.collect()

# Review word profiles and bag of words
user_full_word_counts = {}
global_word_counter = Counter()

if max(WORD_COUNTS) > 0:
    print("Building Bag of Words profiles")
    
    # Tokenizer helper
    def get_tokens(text):
        # Regex for words + lowercase + filter stopwords
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text)]
        return [w for w in words if w not in ENGLISH_STOP_WORDS]

    # Single pass to collect both global stats and user stats
    print("Tokenizing reviews (this takes a moment)...")
    for uid, text in user_reviews.items():
        tokens = get_tokens(text)
        if tokens:
            cnt = Counter(tokens)
            user_full_word_counts[uid] = cnt
            global_word_counter.update(tokens)
            
    del user_reviews
    gc.collect()

# make df
df = pd.DataFrame(interactions)
del interactions
gc.collect()

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

base_df = pd.concat([df_positive, df_negative], ignore_index=True)
base_df = base_df.dropna()

def get_price(gid):
    return games_dict.get(gid, {}).get('price', 0)
base_df['price'] = base_df['item_id'].apply(get_price)

print(f"Base dataset ready: {len(base_df)} rows.")
print("Tags,Words,C,AUC")

# Sweep

# State tracking to avoid re-calculating profiles if N doesn't change
current_tag_n = -1
current_word_n = -1

for t_n in TAG_COUNTS:
    
    # Calculate features (tags)
    if t_n != current_tag_n:
        top_tags = set([t for t, c in global_tag_counter.most_common(t_n)])
        
        # Build lookup dictionary for this N
        active_profiles = {}
        for uid, full_cnt in user_full_tag_counts.items():
            relevant = {k: v for k, v in full_cnt.items() if k in top_tags}
            total = sum(full_cnt.values()) 
            if total > 0 and relevant:
                active_profiles[uid] = {k: v/total for k, v in relevant.items()}
        
        def calculate_tag_affinity(row):
            uid = row['user_id']
            gid = row['item_id']
            if uid not in active_profiles or gid not in games_dict: return 0
            profile = active_profiles[uid]
            game_tags = games_dict[gid]['tags']
            return sum(profile[t] for t in game_tags if t in profile)

        base_df['affinity_score'] = base_df.apply(calculate_tag_affinity, axis=1)
        current_tag_n = t_n

    for w_n in WORD_COUNTS:
        
        # Calculate features (reviews)
        if w_n != current_word_n:
            if w_n > 0:
                top_words = set([w for w, c in global_word_counter.most_common(w_n)])
                
                active_word_profiles = {}
                for uid, full_cnt in user_full_word_counts.items():
                    relevant = {k: v for k, v in full_cnt.items() if k in top_words}
                    total = sum(relevant.values())
                    if total > 0:
                        active_word_profiles[uid] = {k: v/total for k, v in relevant.items()}
                
                def calculate_review_affinity(row):
                    uid = row['user_id']
                    gid = row['item_id']
                    if uid not in active_word_profiles or gid not in games_dict: return 0
                    profile = active_word_profiles[uid]
                    game_tags_lower = games_dict[gid]['tags_lower']
                    return sum(profile[w] for w in game_tags_lower if w in profile)

                base_df['review_affinity_score'] = base_df.apply(calculate_review_affinity, axis=1)
            
            elif w_n == 0 and 'review_affinity_score' in base_df.columns:
                 base_df.drop(columns=['review_affinity_score'], inplace=True)
            
            current_word_n = w_n

        # Training
        features = ['user_items_count', 'item_popularity', 'price', 'affinity_score']
        if w_n > 0:
            features.append('review_affinity_score')

        X = base_df[features]
        y = base_df['target']

        # I decided to set random_state for consistency, someone can tell me if this is normal
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for c_val in C_VALUES:
            model = LogisticRegression(C=c_val, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            y_pred_probability = model.predict_proba(X_test_scaled)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_probability)
            
            # Output CSV format
            print(f"{t_n},{w_n},{c_val},{auc_score:.4f}")
            sys.stdout.flush()