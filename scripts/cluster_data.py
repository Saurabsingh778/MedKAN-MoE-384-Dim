import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import MiniBatchKMeans


# --- CONFIG ---
DATA_FILE = 'kan_deep_data.pt'
OUTPUT_DIR = 'expert_data_splits'
NUM_EXPERTS = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(f"Device: {DEVICE}")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading master dataset...")
data = torch.load(DATA_FILE, weights_only=False)
X = data['inputs'].numpy()
y = data['labels'].numpy()
classes = data['classes']

print(f"Total Data: {X.shape}")
print(f"Total Unique Classes: {len(classes)}")

print("\nRunning Clustering (this determines which expert gets which data)...")
# We use inputs to cluster. 
kmeans = MiniBatchKMeans(n_clusters=NUM_EXPERTS, random_state=42, batch_size=4096, n_init='auto')
expert_ids = kmeans.fit_predict(X)

print(f"Saving splits to {OUTPUT_DIR}/...")
counts = []

for i in range(NUM_EXPERTS):
    mask = (expert_ids == i)
    X_chunk = torch.tensor(X[mask], dtype=torch.float32)
    y_chunk = torch.tensor(y[mask], dtype=torch.long)
    
    save_path = os.path.join(OUTPUT_DIR, f'expert_{i}.pt')
    torch.save({'inputs': X_chunk, 'labels': y_chunk}, save_path)
    
    count = len(X_chunk)
    counts.append(count)
    print(f"  Saved Expert {i}: {count} samples")

# Save the Router Training Data too (we need this later)
torch.save({
    'inputs': torch.tensor(X, dtype=torch.float32), 
    'targets': torch.tensor(expert_ids, dtype=torch.long)
}, 'router_train_data.pt')

plt.bar(range(NUM_EXPERTS), counts)
plt.title("Data Distribution per Expert")
plt.show()