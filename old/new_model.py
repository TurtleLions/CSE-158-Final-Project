import pandas as pd
import ast
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import Counter

print("Generating user/item data")
user_data = []
with open('australian_users_items.json', 'r', encoding='utf-8') as f:
    for line in f:
        user_data.append(ast.literal_eval(line))

print("Generating game metadata")
game_data = []
with open('steam_games.json', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            game_data.append(ast.literal_eval(line))
        except:
            continue

games_dict = {}
all_tags = []

for game in game_data:
    if 'id' in game:
        gid = str(game['id'])
    elif 'id' in game:
        gid = str(game['id'])
    else:
        continue
    
    price = game.get('price', 0)
    # Ensure price makes sense
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
    
    # Combine tags and genres
    combined_tags = list(set(tags + genres))
    all_tags.extend(combined_tags)
    
    games_dict[gid] = {
        'price': price,
        'tags': combined_tags
    }

# Identify top N tags to track, can be changed later
TOP_N = 25
top_tags_counter = Counter(all_tags)
TOP_N_TAGS = [tag for tag, count in top_tags_counter.most_common(TOP_N)]
print(f"Top {TOP_N} tags being tracked: {TOP_N_TAGS}")

print("Creating user affinity profiles")
interactions = []
user_profiles = {}

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
        
        # Record tags if we have the metadata
        if gid in games_dict:
            user_games_tags.extend(games_dict[gid]['tags'])
    # Create user profile
    if user_games_tags:
        total_tags = len(user_games_tags)
        tag_counts = Counter(user_games_tags)
        user_profiles[user_id] = {tag: tag_counts[tag] / total_tags for tag in TOP_N_TAGS if tag in tag_counts}
    else:
        user_profiles[user_id] = {}

df = pd.DataFrame(interactions)
# Only select games that have been played, can be adjusted
df = df[df['playtime_forever'] > 0] 

item_popularity = df.groupby('item_id')['user_id'].nunique().reset_index()
item_popularity.columns = ['item_id', 'item_popularity']
df = df.merge(item_popularity, on='item_id', how='left')

played_pairs = set(zip(df['user_id'], df['item_id']))
all_item_ids = list(df['item_id'].unique())
users = df['user_id'].values
n_samples = len(users)

negative_items = np.empty(n_samples, dtype=object)
filled_mask = np.zeros(n_samples, dtype=bool)

# Safety break to prevent infinite loop if dataset is too dense, ie. cannot find a valid game
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

# Negative feature
df_negative['user_items_count'] = df['user_items_count'].values 
df_negative = df_negative.merge(item_popularity, on='item_id', how='left')

# Pos df
df_positive = df[['user_id', 'item_id', 'user_items_count', 'item_popularity']].copy()
df_positive['target'] = 1

# Together
df_model = pd.concat([df_positive, df_negative], ignore_index=True)
df_model = df_model.dropna()

print("Putting together affinity features")

# Calculation function
def calculate_affinity_score(row):
    uid = row['user_id']
    gid = row['item_id']
    
    # Check existence
    if uid not in user_profiles: return 0
    if gid not in games_dict: return 0
    
    profile = user_profiles[uid]
    game_tags = games_dict[gid]['tags']
    
    # Score = Sum of user's affinity for every tag this game has
    score = 0
    for tag in game_tags:
        if tag in profile:
            score += profile[tag]
            
    return score

df_model['affinity_score'] = df_model.apply(calculate_affinity_score, axis=1)

def get_price(gid):
    return games_dict.get(gid, {}).get('price', 0)

df_model['price'] = df_model['item_id'].apply(get_price)

print("Training Model")
features = ['user_items_count', 'item_popularity', 'price', 'affinity_score']
X = df_model[features]
y = df_model['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_pred_probability = model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

auc_score = roc_auc_score(y_test, y_pred_probability)
print(f"ROC AUC Score: {auc_score:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_probability)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Game Prediction')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve_new.png')

plt.figure(figsize=(10, 5))
sns.barplot(x=features, y=np.abs(model.coef_[0]))
plt.title('Feature Importance (Coefficient Magnitude)')
plt.ylabel('Absolute Coefficient')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance_affinity_new.png')