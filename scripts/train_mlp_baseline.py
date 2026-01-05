import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os, gc

# --- CONFIG ---
DATA_DIR = 'expert_data_splits'
MODEL_DIR = 'trained_models_mlp_small' # New folder
NUM_EXPERTS = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024
LR = 0.005
OUTPUT_DIM = 19756  

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

class MLPExpert(nn.Module):
    """
    Standard MLP Expert with matched parameter budget.
    Structure: Linear -> BatchNorm -> ReLU -> Dropout
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512), # Added for stability at high width
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.final_layer = nn.Linear(256, OUTPUT_DIM)

    def forward(self, x):
        feat = self.net(x)
        return self.final_layer(feat)

def train_expert(expert_id):
    save_path = os.path.join(MODEL_DIR, f'expert_{expert_id}_best.pth')
    if os.path.exists(save_path): return
    print(f"🚀 Training Small MLP Expert {expert_id}...")
    path = os.path.join(DATA_DIR, f'expert_{expert_id}.pt')
    data = torch.load(path, weights_only=False)
    loader = DataLoader(TensorDataset(data['inputs'], data['labels']), batch_size=BATCH_SIZE, shuffle=True)
    
    model = MLPExpert().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    for epoch in range(100): # Fast convergence
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({'state_dict': model.state_dict()}, save_path)
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    dummy = MLPExpert()
    print(f"MLP Parameters: {sum(p.numel() for p in dummy.parameters()):,}")
    for i in range(NUM_EXPERTS): train_expert(i)