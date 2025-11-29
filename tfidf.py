import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
import ast
import gzip
import os
import sys
import gc
import re
from collections import defaultdict

MAX_TEXT_FEATURES = 3000
MAX_TAG_FEATURES = 100
EMBEDDING_DIM = 64
BATCH_SIZE = 256
EPOCHS = 5

def smart_open(file_path, mode='rt', encoding='utf-8'):
    if os.path.exists(file_path):
        if file_path.endswith('.gz'):
            return gzip.open(file_path, mode=mode, encoding=encoding)
        return open(file_path, mode=mode, encoding=encoding)
    
    gz_path = file_path + '.gz'
    if os.path.exists(gz_path):
        return gzip.open(gz_path, mode=mode, encoding=encoding)
    return open(file_path, mode=mode, encoding=encoding)

def clean_text(text):
    if not text: return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text

print("Loading data")

# Load User/Item Interactions
print("Loading interactions")
user_data = []
with smart_open('australian_users_items.json.gz') as f:
    for line in f:
        user_data.append(ast.literal_eval(line))

# Load Game Metadata
print("Loading game metadata")
games_dict = {}
with smart_open('steam_games.json.gz') as f:
    for line in f:
        try:
            game = ast.literal_eval(line)
            gid = None
            if 'id' in game: gid = str(game['id'])
            elif 'app_id' in game: gid = str(game['app_id'])
            if not gid: continue
            
            # Extract Price
            price = game.get('price', 0)
            if isinstance(price, str):
                price = 0 if 'free' in price.lower() else float(price) if price.replace('.', '', 1).isdigit() else 0
            
            # Extract Tags
            tags = game.get('tags', []) + game.get('genres', [])
            tags_str = " ".join([t.lower() for t in tags])
            
            games_dict[gid] = {'price': price, 'tags': tags_str}
        except:
            continue

# Load Reviews
print("Loading reviews")
user_reviews_text = defaultdict(list)
with smart_open('steam_reviews.json.gz') as f:
    for line in f:
        try:
            node = ast.literal_eval(line)
            # Resolve ID
            uid = str(node.get('user_id', node.get('username', '')))
            if not uid: continue
            
            text = node.get('text', '')
            if text:
                user_reviews_text[uid].append(clean_text(text))
        except:
            continue

# Collapse reviews into single string per user
user_reviews_map = {uid: " ".join(texts) for uid, texts in user_reviews_text.items()}
del user_reviews_text
gc.collect()

print("df creation")

interactions = []
for user in user_data:
    uid = str(user['user_id'])
    for item in user['items']:
        interactions.append({
            'user_id': uid,
            'item_id': str(item['item_id']),
            'playtime': item['playtime_forever']
        })

df = pd.DataFrame(interactions)
del user_data, interactions
gc.collect()

# Filter only played games and existing games
df = df[df['playtime'] > 0]
df = df[df['item_id'].isin(games_dict.keys())]

# Calculate popularity
pop_counts = df['item_id'].value_counts().to_dict()
df['popularity'] = df['item_id'].map(pop_counts)

# Negative sampling
print("Creating negative samples")
# Get list of all users and top 500 popular items
all_users = df['user_id'].unique()
top_items = list(df['item_id'].value_counts().head(500).index)

# Create a set of existing pairs for fast lookup
existing_pairs = set(zip(df['user_id'], df['item_id']))

n_positives = len(df)
target_negatives = n_positives

unique_users = df['user_id'].unique()
top_items_array = np.array(top_items)
existing_pairs = set(zip(df['user_id'], df['item_id']))

negative_rows = []

# Generate batches until we hit the target
while len(negative_rows) < target_negatives:
    needed = target_negatives - len(negative_rows)
    
    batch_size = int(needed * 1.2)
    
    batch_u = np.random.choice(unique_users, size=batch_size)
    batch_i = np.random.choice(top_items_array, size=batch_size)
    
    # Filter valid negatives
    for u, i in zip(batch_u, batch_i):
        if len(negative_rows) >= target_negatives:
            break
            
        if (u, i) not in existing_pairs:
            negative_rows.append({
                'user_id': u,
                'item_id': i,
                'playtime': 0,
                'popularity': pop_counts.get(i, 0)
            })
            # Add to existing_pairs to ensure we don't accidentally add the same negative twice
            existing_pairs.add((u, i))

df_neg = pd.DataFrame(negative_rows)
df_neg['target'] = 0
df['target'] = 1

# Merge Positive and Negative
full_df = pd.concat([df[['user_id', 'item_id', 'popularity', 'target']], 
                     df_neg[['user_id', 'item_id', 'popularity', 'target']]], ignore_index=True)

# Shuffle
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset Size: {len(full_df)} interactions")

print("Features")

# ID Encoding (Label Encoding)
print("Encoding IDs")
user_encoder = LabelEncoder()
full_df['user_idx'] = user_encoder.fit_transform(full_df['user_id'])
n_users = len(user_encoder.classes_)

item_encoder = LabelEncoder()
full_df['item_idx'] = item_encoder.fit_transform(full_df['item_id'])
n_items = len(item_encoder.classes_)

# TF-IDF on User Reviews
print("Vectorizing reviews")
reviews_series = full_df['user_id'].map(lambda u: user_reviews_map.get(u, ""))
tfidf_rev = TfidfVectorizer(max_features=MAX_TEXT_FEATURES, stop_words='english')
X_reviews = tfidf_rev.fit_transform(reviews_series).toarray().astype(np.float32)

# TF-IDF on Game Tags
print("Vectorizing game tags")
tags_series = full_df['item_id'].map(lambda i: games_dict.get(i, {}).get('tags', ""))
tfidf_tags = TfidfVectorizer(max_features=MAX_TAG_FEATURES, stop_words='english')
X_tags = tfidf_tags.fit_transform(tags_series).toarray().astype(np.float32)

# Numerical Features (Price & Popularity)
print("Normalizing Numerical Features")
price_series = full_df['item_id'].map(lambda i: games_dict.get(i, {}).get('price', 0)).values.reshape(-1, 1)
pop_series = full_df['popularity'].values.reshape(-1, 1)

scaler = StandardScaler()
X_nums = scaler.fit_transform(np.hstack([price_series, pop_series])).astype(np.float32)

X_side = np.hstack([X_reviews, X_tags, X_nums])

print(f"Final Side Feature Dimension: {X_side.shape[1]}")

# Clean up memory
del reviews_series, tags_series, X_reviews, X_tags, X_nums
gc.collect()

# Prepare inputs
X_u = full_df['user_idx'].values
X_i = full_df['item_idx'].values
y = full_df['target'].values

# Train/Test split
X_u_train, X_u_test, X_i_train, X_i_test, X_s_train, X_s_test, y_train, y_test = train_test_split(
    X_u, X_i, X_side, y, test_size=0.2, random_state=42
)

print("Generating model")

# Define inputs
user_input = layers.Input(shape=(1,), name='user_input')
item_input = layers.Input(shape=(1,), name='item_input')
side_input = layers.Input(shape=(X_side.shape[1],), name='side_input')

# Latent factor embeddings
u_emb = layers.Embedding(n_users, EMBEDDING_DIM, embeddings_regularizer=regularizers.l2(1e-6))(user_input)
i_emb = layers.Embedding(n_items, EMBEDDING_DIM, embeddings_regularizer=regularizers.l2(1e-6))(item_input)

u_vec = layers.Flatten()(u_emb)
i_vec = layers.Flatten()(i_emb)

mf_layer = layers.Dot(axes=1)([u_vec, i_vec])

dense_1 = layers.Dense(256, activation='relu')(side_input)
dense_1 = layers.Dropout(0.4)(dense_1)
dense_2 = layers.Dense(128, activation='relu')(dense_1)

# Include the raw user/item vectors in the deep path too, to learn non-linear interactions
concat = layers.Concatenate()([mf_layer, u_vec, i_vec, dense_2])

# Final prediction layer
pred_layer = layers.Dense(64, activation='relu')(concat)
output = layers.Dense(1, activation='sigmoid')(pred_layer)

model = Model(inputs=[user_input, item_input, side_input], outputs=output)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision')]
)

model.summary()

print("Training")

history = model.fit(
    [X_u_train, X_i_train, X_s_train],
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=([X_u_test, X_i_test, X_s_test], y_test),
    verbose=1
)

print("Finished training")

preds = model.predict([X_u_test, X_i_test, X_s_test], batch_size=BATCH_SIZE)
roc = roc_auc_score(y_test, preds)
pr = average_precision_score(y_test, preds)

print(f"ROC AUC: {roc:.4f}")
print(f"PR AUC:  {pr:.4f}")

# Example Predictions
print("\nSample Predictions:")
for i in range(5):
    p = preds[i][0]
    actual = y_test[i]
    print(f"Predicted: {p:.4f} | Actual: {actual}")