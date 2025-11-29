import pandas as pd
import ast
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, roc_curve
from sklearn.preprocessing import StandardScaler
import seaborn as sns

print("Loading data")
data = []
with open('australian_users_items.json', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(ast.literal_eval(line))

interactions = []
for user in data:
    uid = user['user_id']
    u_items = user['items_count']
    for item in user['items']:
        interactions.append({
            'user_id': uid,
            'item_id': item['item_id'],
            'playtime_forever': item['playtime_forever'],
            'user_items_count': u_items
        })

df = pd.DataFrame(interactions)
df_positive = df[df['playtime_forever'] > 0].copy()
df_positive['target'] = 1

# Neg sampling
print("Generating neg samples")

top_games = df_positive['item_id'].value_counts().head(500).index.tolist()

played_pairs = set(zip(df_positive['user_id'], df_positive['item_id']))
users = df_positive['user_id'].values
n_samples = len(users)

negative_items = np.empty(n_samples, dtype=object)
filled_mask = np.zeros(n_samples, dtype=bool)
retry_count = 0

while not filled_mask.all() and retry_count < 100:
    missing = (~filled_mask).sum()
    # Sample only from Top 500 games
    candidates = np.random.choice(top_games, size=missing)
    current_users = users[~filled_mask]
    
    valid_mask = np.array([(u, i) not in played_pairs for u, i in zip(current_users, candidates)])
    
    if valid_mask.any():
        idx = np.where(~filled_mask)[0][valid_mask]
        negative_items[idx] = candidates[valid_mask]
        filled_mask[idx] = True
    retry_count += 1
    if retry_count % 10 == 0:
        print(f"Pass {retry_count}: {filled_mask.sum()}/{n_samples}")

df_negative = pd.DataFrame({
    'user_id': users,
    'item_id': negative_items,
    'user_items_count': df_positive['user_items_count'].values,
    'target': 0
})

# Combine
df_model = pd.concat([df_positive[['user_id', 'item_id', 'user_items_count', 'target']], 
                      df_negative[['user_id', 'item_id', 'user_items_count', 'target']]], ignore_index=True)
df_model = df_model.dropna().sample(frac=1, random_state=42).reset_index(drop=True)

# Train/Test Split
train_df, test_df = train_test_split(df_model, test_size=0.2, random_state=42)

# Feature Engineering
print("Calculating features")
train_pos = train_df[train_df['target'] == 1]
pop_map = train_pos.groupby('item_id')['user_id'].nunique().reset_index()
pop_map.columns = ['item_id', 'item_popularity']

train_df = train_df.merge(pop_map, on='item_id', how='left').fillna(0)
test_df = test_df.merge(pop_map, on='item_id', how='left').fillna(0)

features = ['user_items_count', 'item_popularity']
X_train = train_df[features]
y_train = train_df['target']
X_test = test_df[features]
y_test = test_df['target']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Model")
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_probs = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)

print(f"ROC AUC: {auc:.4f}")
print(f"PR AUC:  {pr_auc:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# plotting

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Game Prediction')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve_base.png')

plt.figure(figsize=(10, 5))
sns.barplot(x=features, y=np.abs(model.coef_[0]))
plt.title('Feature Importance (Coefficient Magnitude)')
plt.ylabel('Absolute Coefficient')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance_base.png')