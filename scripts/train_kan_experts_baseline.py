import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
import time
import numpy as np
import gc
from src.moe_kan_lib import KANNetwork, MoEKAN

# --- CONFIGURATION ---
DATA_DIR = 'expert_data_splits'
MODEL_DIR = 'trained_models'
LOG_FILE = 'training_log.txt'
NUM_EXPERTS = 32
KAN_LAYERS = [384, 256, 128]
OUTPUT_DIM = 19756         
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training Hyperparameters
MAX_EPOCHS = 500        
PATIENCE = 15             
BATCH_SIZE = 1024
LR = 0.005
TARGET_ACCURACY = 0.98  # Exit if 98% is reached
MIN_DELTA = 0.0001     # Minimum improvement required to stay in training (0.01%)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger(LOG_FILE)

def load_balancing_loss(probs):
    """Auxiliary loss to prevent expert collapse."""
    importance = probs.mean(dim=0)
    loss = torch.std(importance) / (importance.mean() + 1e-9)
    return loss

def train_expert(expert_id):
    save_path = os.path.join(MODEL_DIR, f'expert_{expert_id}_best.pth')
    
    if os.path.exists(save_path):
        print(f"✅ Expert {expert_id} already trained. Skipping...")
        return

    print(f"\n" + "="*40)
    print(f"🚀 TRAINING EXPERT {expert_id}/{NUM_EXPERTS-1}")
    print("="*40)

    # 1. Load Data
    path = os.path.join(DATA_DIR, f'expert_{expert_id}.pt')
    if not os.path.exists(path):
        return

    data = torch.load(path, weights_only=False)
    inputs, labels = data['inputs'], data['labels']
    
    if len(labels) == 0: return

    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Architecture
    model = KANNetwork(KAN_LAYERS, grid_size=5, spline_order=3).to(DEVICE)
    final_layer = nn.Linear(KAN_LAYERS[-1], OUTPUT_DIM).to(DEVICE)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': final_layer.parameters()}
    ], lr=LR)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        final_layer.train()
        
        running_loss = 0.0
        running_acc = 0.0
        batches = 0
        
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            
            features = model(bx)
            logits = final_layer(features)
            loss = criterion(logits, by)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            acc = (torch.argmax(logits, dim=1) == by).float().mean().item()
            running_loss += loss.item()
            running_acc += acc
            batches += 1
            
        avg_loss = running_loss / batches
        avg_acc = running_acc / batches
        
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Ep {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # --- UPDATED EXIT LOGIC ---
        
        # 1. Target Accuracy Exit
        if avg_acc >= TARGET_ACCURACY:
            print(f"  🎯 Target Accuracy ({TARGET_ACCURACY*100}%) reached. Saving and exiting.")
            torch.save({
                'expert_state': model.state_dict(),
                'final_state': final_layer.state_dict(),
                'stats': {'acc': avg_acc, 'loss': avg_loss}
            }, save_path)
            break

        # 2. Early Stopping & Checkpointing (Loss-based)
        if avg_loss < (best_loss - MIN_DELTA):
            best_loss = avg_loss
            patience_counter = 0
            # Always save the absolute best state
            torch.save({
                'expert_state': model.state_dict(),
                'final_state': final_layer.state_dict(),
                'stats': {'acc': avg_acc, 'loss': avg_loss}
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  🛑 Early stopping: No significant improvement in {PATIENCE} epochs.")
                break

    # Cleanup GPU memory explicitly
    del model, final_layer, optimizer, loader, dataset, inputs, labels
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    print(f"Logging started at {time.ctime()}")
    start_time = time.time()
    
    if not os.path.exists('router_weights.pth'):
        print("⚠️ Warning: 'router_weights.pth' not found. Ensure routing is pre-calculated.")

    for i in range(NUM_EXPERTS):
        try:
            train_expert(i)
        except Exception as e:
            if "CUDA" in str(e) or "memory" in str(e):
                print(f"\n❌ GPU ERROR on Expert {i}: {e}")
                sys.exit(1)
            else:
                print(f"❌ Error training Expert {i}: {e}")
            
    total_time = (time.time() - start_time) / 3600
    print(f"\n🎉 ALL DONE! Total time: {total_time:.2f} hours.")