import pandas as pd
import ast
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, average_precision_score # Added average_precision_score
import seaborn as sns

print("Gathering data")
data = []
with open('australian_users_items.json', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(ast.literal_eval(line))

interactions = []
for user in data:
    user_id = user['user_id']
    items_count = user['items_count']
    for item in user['items']:
        interactions.append({
            'user_id': user_id,
            'item_id': item['item_id'],
            'playtime_forever': item['playtime_forever'],
            'user_items_count': items_count
        })

df = pd.DataFrame(interactions)

# Only select games that have been played, can be adjusted
df = df[df['playtime_forever'] > 0]

# Checking popularity of pos

# 50/50 pos/neg sampling
played_pairs = set(zip(df['user_id'], df['item_id']))
all_item_ids = df['item_id'].unique()
users = df['user_id'].values
n_samples = len(users)

negative_items = np.empty(n_samples, dtype=object)
filled_mask = np.zeros(n_samples, dtype=bool)

iter_count = 0
# Loop to generate a correct number of samples
while not filled_mask.all():
    iter_count += 1
    missing_count = (~filled_mask).sum()
    candidates = np.random.choice(all_item_ids, size=missing_count)
    current_users = users[~filled_mask]
    valid_mask = np.array([ (u, i) not in played_pairs for u, i in zip(current_users, candidates) ])
    missing_indices = np.where(~filled_mask)[0]
    
    if valid_mask.any():
        update_indices = missing_indices[valid_mask]
        negative_items[update_indices] = candidates[valid_mask]
        filled_mask[update_indices] = True
        
    print(f"Pass {iter_count}: has {filled_mask.sum()}/{n_samples} samples completed")

df_negative = pd.DataFrame({
    'user_id': users,
    'item_id': negative_items,
    'target': 0
})

# Negative feature
df_negative['user_items_count'] = df['user_items_count'].values

# Pos df
df_positive = df[['user_id', 'item_id', 'user_items_count']].copy()
df_positive['target'] = 1

# Together
df_model = pd.concat([df_positive, df_negative], ignore_index=True)

# Shuffle
df_model = df_model.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Final dataset size: {len(df_model)}")
print(f"Checking class balance:\n{df_model['target'].value_counts()}")

print("Training")

# Bad data handling
df_model = df_model.dropna()

# Split to 80% train and 20% test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# New split logic:
train_df, test_df = train_test_split(df_model, test_size=0.2, random_state=42)

print("Calculating popularity on train set")
# Filter for positives in train to count popularity
train_positives = train_df[train_df['target'] == 1]
popularity_map = train_positives.groupby('item_id')['user_id'].nunique().reset_index()
popularity_map.columns = ['item_id', 'item_popularity']

# Map popularity to train set
train_df = train_df.merge(popularity_map, on='item_id', how='left')
train_df['item_popularity'] = train_df['item_popularity'].fillna(0) 

# Map popularity to test set 
test_df = test_df.merge(popularity_map, on='item_id', how='left')
test_df['item_popularity'] = test_df['item_popularity'].fillna(0) 

features = ['user_items_count', 'item_popularity']
X_train = train_df[features]
y_train = train_df['target']
X_test = test_df[features]
y_test = test_df['target']

# Logistic reg classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Eval steps
y_pred = model.predict(X_test)
y_pred_probability = model.predict_proba(X_test)[:, 1]

# Print Metrics
print("\nClassification Report from sklearn (it looks cool so i like it):")
print(classification_report(y_test, y_pred))

# Receiver Operating Characteristic Area Under the Curve (basically just a metric to determine acc and Type 1/Type 2 error)
auc_score = roc_auc_score(y_test, y_pred_probability)
print(f"ROC AUC Score: {auc_score:.4f}")

# Precision-Recall AUC is better for imbalanced or recommender tasks apparently
pr_auc = average_precision_score(y_test, y_pred_probability)
print(f"PR AUC Score (Average Precision): {pr_auc:.4f}")
# ------------------------

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probability)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Game Prediction')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve_base.png')

plt.figure(figsize=(8, 4))
sns.barplot(x=features, y=model.coef_[0])
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.ylabel('Coefficient Magnitude')
plt.tight_layout()
plt.savefig('feature_importance_base.png')