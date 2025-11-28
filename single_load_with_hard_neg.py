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
from sklearn.metrics import roc_auc_score, average_precision_score # Added average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import multiprocessing

TAG_COUNTS = [5, 10, 15, 20, 25, 30, 40] 
WORD_COUNTS = [50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
C_VALUES = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
CORES = 10

def smart_open(file_path, mode='rt', encoding='utf-8'):
    if os.path.exists(file_path):
        if file_path.endswith('.gz'):
            return gzip.open(file_path, mode=mode, encoding=encoding)
        return open(file_path, mode=mode, encoding=encoding)
    
    gz_path = file_path + '.gz'
    if os.path.exists(gz_path):
        return gzip.open(gz_path, mode=mode, encoding=encoding)

    return open(file_path, mode=mode, encoding=encoding)

def get_tokens(text):
    words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text)]
    return [w for w in words if w not in ENGLISH_STOP_WORDS]

print("Loading user/item interactions", file=sys.stderr)
user_data = []
try:
    with smart_open('australian_users_items.json.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            user_data.append(ast.literal_eval(line))
except FileNotFoundError:
    print("Error: australian_users_items.json.gz not found.", file=sys.stderr)
    sys.exit(1)

print("Loading game metadata", file=sys.stderr)
game_data = []
try:
    with smart_open('steam_games.json.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                game_data.append(ast.literal_eval(line))
            except:
                continue
except FileNotFoundError:
    print("Error: steam_games.json.gz not found.", file=sys.stderr)
    sys.exit(1)

# Load reviews
user_reviews = {}
if max(WORD_COUNTS) > 0:
    print("Loading user reviews", file=sys.stderr)
    try:
        with smart_open('australian_user_reviews.json.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                review_node = ast.literal_eval(line)
                user_id = str(review_node['user_id'])
                full_text = " ".join([r.get('review', '') for r in review_node.get('reviews', [])])
                user_reviews[user_id] = full_text
    except FileNotFoundError:
        print("Warning: Reviews file not found. Skipping text features.", file=sys.stderr)
        WORD_COUNTS = [0]

# Process metadata
print("Processing Metadata", file=sys.stderr)
games_dict = {}
all_tags_global = []

for game in game_data:
    gid = None
    if 'id' in game: gid = str(game['id'])
    elif 'app_id' in game: gid = str(game['app_id'])
    
    if not gid: continue
    
    price = game.get('price', 0)
    if isinstance(price, str):
        if 'free' in price.lower(): price = 0
        else:
            try: price = float(price)
            except: price = 0
                
    tags = game.get('tags', [])
    genres = game.get('genres', [])
    
    combined_tags = list(set(tags + genres))
    normalized_tags = [t.lower() for t in combined_tags]
    all_tags_global.extend(combined_tags)
    
    games_dict[gid] = {
        'price': price,
        'tags': combined_tags,
        'tags_lower': set(normalized_tags)
    }

# Tag profiles
global_tag_counter = Counter(all_tags_global)
del game_data 
del all_tags_global

interactions = []
user_full_tag_counts = {} 

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

# Review word profiles
user_full_word_counts = {}
global_word_counter = Counter()

if max(WORD_COUNTS) > 0:
    print("Tokenizing reviews", file=sys.stderr)
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
print("Creating Negative Samples", file=sys.stderr)
played_pairs = set(zip(df['user_id'], df['item_id']))
all_item_ids = list(df['item_id'].unique())

# Identify top 500 popular games for hard negatives
top_games = df['item_id'].value_counts().head(500).index.tolist()

users = df['user_id'].values
n_samples = len(users)

negative_items = np.empty(n_samples, dtype=object)
filled_mask = np.zeros(n_samples, dtype=bool)
retry_count = 0

while not filled_mask.all() and retry_count < 100:
    missing_count = (~filled_mask).sum()
    
    candidates = np.random.choice(top_games, size=missing_count)
    current_users = users[~filled_mask]
    
    valid_mask = np.array([ (u, i) not in played_pairs for u, i in zip(current_users, candidates) ])
    if valid_mask.any():
        update_indices = np.where(~filled_mask)[0][valid_mask]
        negative_items[update_indices] = candidates[valid_mask]
        filled_mask[update_indices] = True
    retry_count += 1

df_negative = pd.DataFrame({'user_id': users, 'item_id': negative_items, 'target': 0})
df_negative['user_items_count'] = df['user_items_count'].values 
df_negative = df_negative.merge(item_popularity, on='item_id', how='left')

df_positive = df[['user_id', 'item_id', 'user_items_count', 'item_popularity']].copy()
df_positive['target'] = 1

base_df = pd.concat([df_positive, df_negative], ignore_index=True).dropna()

def get_price(gid):
    return games_dict.get(gid, {}).get('price', 0)
base_df['price'] = base_df['item_id'].apply(get_price)

print(f"Base dataset ready: {len(base_df)} rows.", file=sys.stderr)

# Trains model on 1 grid point
def process_job(params):
    t_n, w_n, c_val = params
    
    # Copy dataset subset
    work_df = base_df[['user_id', 'item_id', 'user_items_count', 'item_popularity', 'price', 'target']].copy()

    # Calculate tag affinity
    top_tags = set([t for t, c in global_tag_counter.most_common(t_n)])
    
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

    work_df['affinity_score'] = work_df.apply(calculate_tag_affinity, axis=1)

    # Calculate review affinity
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

        work_df['review_affinity_score'] = work_df.apply(calculate_review_affinity, axis=1)

    # Training
    features = ['user_items_count', 'item_popularity', 'price', 'affinity_score']
    if w_n > 0:
        features.append('review_affinity_score')

    X = work_df[features]
    y = work_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(C=c_val, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    # Calculate PR AUC
    pr_auc = average_precision_score(y_test, probs)
    
    return f"{t_n},{w_n},{c_val},{auc:.4f},{pr_auc:.4f}"

if __name__ == '__main__':
    param_list = [(t, w, c) for t in TAG_COUNTS for w in WORD_COUNTS for c in C_VALUES]
    
    print(f"Starting sweep of {len(param_list)} jobs using {CORES} cores...", file=sys.stderr)
    
    # Updated header to include PR_AUC
    print("Tags,Words,C,ROC_AUC,PR_AUC")
    
    with multiprocessing.Pool(processes=CORES) as pool:
        for result in pool.imap_unordered(process_job, param_list):
            print(result)
            sys.stdout.flush()