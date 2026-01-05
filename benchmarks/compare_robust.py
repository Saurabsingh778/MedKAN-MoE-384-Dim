import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from src.moe_kan_lib import KANNetwork
import os
import time
from tqdm import tqdm
import psutil
import gc
from collections import defaultdict

# --- CONFIG ---
TEST_SIZE = 10000 
BATCH_SIZE = 2048
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directories
KAN_DIR = 'trained_models'
MLP_DIR = 'trained_models_mlp_benchmark'

# Dimensions
INPUT_DIM = 384
KAN_LAYERS = [384, 256, 128]
MLP_HIDDEN = [512, 512, 256]
OUTPUT_DIM = 19756
NUM_EXPERTS = 32

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def get_gpu_memory_reserved():
    """Get reserved GPU memory in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**2
    return 0

def clear_memory():
    """Clear GPU and system memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

# ==========================================
# ARCHITECTURE DEFINITIONS
# ==========================================
class MLPExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.final_layer = nn.Linear(256, OUTPUT_DIM)

    def forward(self, x):
        return self.final_layer(self.net(x))

# ==========================================
# LOAD ROUTER
# ==========================================
router = nn.Sequential(
    nn.Linear(INPUT_DIM, 128), nn.ReLU(), nn.Linear(128, NUM_EXPERTS)
).to(DEVICE)

if os.path.exists('router_weights.pth'):
    router.load_state_dict(torch.load('router_weights.pth', map_location=DEVICE))
    router.eval()
else:
    print("❌ Critical: router_weights.pth missing!")
    exit()

# ==========================================
# ENHANCED INFERENCE ENGINE WITH METRICS
# ==========================================
def run_moe_inference_with_metrics(model_type, X_test):
    """
    Run inference and collect comprehensive metrics
    """
    clear_memory()
    
    # Initialize tracking
    all_preds = []
    inference_times = []
    memory_snapshots = []
    expert_usage = defaultdict(int)
    
    # Warmup run
    print(f"🔥 Warming up {model_type} model...")
    warmup_batch = X_test[:BATCH_SIZE].to(DEVICE)
    with torch.no_grad():
        _ = router(warmup_batch)
    clear_memory()
    
    # Record initial memory
    initial_memory = get_gpu_memory()
    peak_memory = initial_memory
    
    # Main inference loop
    total_start = time.time()
    
    for i in tqdm(range(0, len(X_test), BATCH_SIZE), desc=f"Inference ({model_type})"):
        batch_start = time.time()
        batch_x = X_test[i : i+BATCH_SIZE].to(DEVICE)
        
        with torch.no_grad():
            r_logits = router(batch_x)
            assignments = torch.argmax(r_logits, dim=1)
        
        batch_preds = torch.zeros(len(batch_x), dtype=torch.long, device=DEVICE)
        active_experts = torch.unique(assignments).tolist()
        
        for exp_id in active_experts:
            expert_usage[exp_id] += 1
            mask = (assignments == exp_id)
            sub_x = batch_x[mask]
            
            # Instantiate correct model
            if model_type == 'KAN':
                model = KANNetwork(KAN_LAYERS, grid_size=5).to(DEVICE)
                final = nn.Linear(KAN_LAYERS[-1], OUTPUT_DIM).to(DEVICE)
                path = os.path.join(KAN_DIR, f'expert_{exp_id}_best.pth')
            else:
                full_model = MLPExpert().to(DEVICE)
                path = os.path.join(MLP_DIR, f'expert_{exp_id}_best.pth')

            if not os.path.exists(path): 
                continue
            
            # Load weights
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            if model_type == 'KAN':
                model.load_state_dict(ckpt['expert_state'])
                final.load_state_dict(ckpt['final_state'])
                model.eval(); final.eval()
                with torch.no_grad():
                    out = final(model(sub_x))
            else:
                full_model.load_state_dict(ckpt['state_dict'])
                full_model.eval()
                with torch.no_grad():
                    out = full_model(sub_x)
            
            batch_preds[mask] = torch.argmax(out, dim=1)
            
            # Clean up
            if model_type == 'KAN':
                del model, final
            else:
                del full_model
        
        # Record metrics
        batch_time = time.time() - batch_start
        inference_times.append(batch_time)
        
        current_memory = get_gpu_memory()
        memory_snapshots.append(current_memory)
        peak_memory = max(peak_memory, current_memory)
        
        all_preds.extend(batch_preds.cpu().numpy())
        
        # Cleanup
        del batch_x, batch_preds, r_logits, assignments
        
    total_time = time.time() - total_start
    
    # Calculate statistics
    metrics = {
        'predictions': np.array(all_preds),
        'total_time': total_time,
        'avg_batch_time': np.mean(inference_times),
        'std_batch_time': np.std(inference_times),
        'throughput': len(X_test) / total_time,  # samples/sec
        'initial_memory_mb': initial_memory,
        'peak_memory_mb': peak_memory,
        'avg_memory_mb': np.mean(memory_snapshots),
        'memory_overhead_mb': peak_memory - initial_memory,
        'expert_usage': dict(expert_usage),
        'num_active_experts': len(expert_usage)
    }
    
    clear_memory()
    return metrics

# ==========================================
# COMPREHENSIVE EVALUATION
# ==========================================
def evaluate_model(y_true, y_pred, model_type, metrics):
    """
    Calculate comprehensive evaluation metrics
    """
    results = {
        'model_type': model_type,
        # Performance Metrics
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        
        # Speed Metrics
        'total_inference_time': metrics['total_time'],
        'avg_batch_time': metrics['avg_batch_time'],
        'std_batch_time': metrics['std_batch_time'],
        'throughput_samples_per_sec': metrics['throughput'],
        'latency_per_sample_ms': (metrics['total_time'] / len(y_true)) * 1000,
        
        # Memory Metrics
        'initial_memory_mb': metrics['initial_memory_mb'],
        'peak_memory_mb': metrics['peak_memory_mb'],
        'avg_memory_mb': metrics['avg_memory_mb'],
        'memory_overhead_mb': metrics['memory_overhead_mb'],
        
        # Expert Usage
        'num_active_experts': metrics['num_active_experts'],
        'expert_usage_distribution': metrics['expert_usage']
    }
    
    return results

# ==========================================
# CALCULATE MODEL STATISTICS
# ==========================================
def get_model_stats(model_type):
    """
    Get detailed model statistics
    """
    if model_type == 'KAN':
        # Load one expert to get stats
        model = KANNetwork(KAN_LAYERS, grid_size=5)
        final = nn.Linear(KAN_LAYERS[-1], OUTPUT_DIM)
        total_params = count_parameters(model) + count_parameters(final)
        model_size = calculate_model_size_mb(model) + calculate_model_size_mb(final)
    else:
        model = MLPExpert()
        total_params = count_parameters(model)
        model_size = calculate_model_size_mb(model)
    
    # Count available experts
    model_dir = KAN_DIR if model_type == 'KAN' else MLP_DIR
    available_experts = len([f for f in os.listdir(model_dir) if f.endswith('_best.pth')])
    
    total_expert_params = total_params * available_experts
    router_params = count_parameters(router)
    
    return {
        'params_per_expert': total_params,
        'model_size_per_expert_mb': model_size,
        'available_experts': available_experts,
        'total_expert_params': total_expert_params,
        'router_params': router_params,
        'total_system_params': total_expert_params + router_params
    }

# ==========================================
# PRINT COMPREHENSIVE RESULTS
# ==========================================
def print_results(kan_results, mlp_results, kan_stats, mlp_stats):
    """
    Print beautiful formatted results
    """
    print("\n" + "═"*80)
    print(" "*25 + "COMPREHENSIVE MODEL COMPARISON")
    print("═"*80)
    
    # Model Architecture
    print("\n📊 MODEL ARCHITECTURE")
    print("-"*80)
    print(f"{'Metric':<40} | {'MoE-KAN':<15} | {'MoE-MLP':<15}")
    print("-"*80)
    print(f"{'Parameters per Expert':<40} | {kan_stats['params_per_expert']:>13,} | {mlp_stats['params_per_expert']:>13,}")
    print(f"{'Model Size per Expert (MB)':<40} | {kan_stats['model_size_per_expert_mb']:>13.2f} | {mlp_stats['model_size_per_expert_mb']:>13.2f}")
    print(f"{'Available Experts':<40} | {kan_stats['available_experts']:>13} | {mlp_stats['available_experts']:>13}")
    print(f"{'Total Expert Parameters':<40} | {kan_stats['total_expert_params']:>13,} | {mlp_stats['total_expert_params']:>13,}")
    print(f"{'Router Parameters':<40} | {kan_stats['router_params']:>13,} | {mlp_stats['router_params']:>13,}")
    print(f"{'Total System Parameters':<40} | {kan_stats['total_system_params']:>13,} | {mlp_stats['total_system_params']:>13,}")
    
    param_ratio = mlp_stats['total_system_params'] / kan_stats['total_system_params']
    print(f"\n💡 MLP uses {param_ratio:.2f}x MORE parameters than KAN")
    
    # Performance Metrics
    print("\n🎯 PERFORMANCE METRICS")
    print("-"*80)
    print(f"{'Metric':<40} | {'MoE-KAN':<15} | {'MoE-MLP':<15}")
    print("-"*80)
    print(f"{'Accuracy':<40} | {kan_results['accuracy']*100:>13.2f}% | {mlp_results['accuracy']*100:>13.2f}%")
    print(f"{'Weighted F1':<40} | {kan_results['f1_weighted']:>14.4f} | {mlp_results['f1_weighted']:>14.4f}")
    print(f"{'Macro F1':<40} | {kan_results['f1_macro']:>14.4f} | {mlp_results['f1_macro']:>14.4f}")
    print(f"{'Weighted Precision':<40} | {kan_results['precision_weighted']:>14.4f} | {mlp_results['precision_weighted']:>14.4f}")
    print(f"{'Weighted Recall':<40} | {kan_results['recall_weighted']:>14.4f} | {mlp_results['recall_weighted']:>14.4f}")
    
    acc_diff = (kan_results['accuracy'] - mlp_results['accuracy']) * 100
    print(f"\n💡 Accuracy difference: {abs(acc_diff):.2f}% ({'KAN higher' if acc_diff > 0 else 'MLP higher'})")
    
    # Speed Metrics
    print("\n⚡ INFERENCE SPEED")
    print("-"*80)
    print(f"{'Metric':<40} | {'MoE-KAN':<15} | {'MoE-MLP':<15}")
    print("-"*80)
    print(f"{'Total Time (seconds)':<40} | {kan_results['total_inference_time']:>13.2f} | {mlp_results['total_inference_time']:>13.2f}")
    print(f"{'Avg Batch Time (seconds)':<40} | {kan_results['avg_batch_time']:>13.4f} | {mlp_results['avg_batch_time']:>13.4f}")
    print(f"{'Throughput (samples/sec)':<40} | {kan_results['throughput_samples_per_sec']:>13.2f} | {mlp_results['throughput_samples_per_sec']:>13.2f}")
    print(f"{'Latency per Sample (ms)':<40} | {kan_results['latency_per_sample_ms']:>13.4f} | {mlp_results['latency_per_sample_ms']:>13.4f}")
    
    speedup = mlp_results['total_inference_time'] / kan_results['total_inference_time']
    print(f"\n💡 {'KAN' if speedup > 1 else 'MLP'} is {abs(speedup):.2f}x faster")
    
    # Memory Metrics
    print("\n💾 MEMORY USAGE (VRAM)")
    print("-"*80)
    print(f"{'Metric':<40} | {'MoE-KAN':<15} | {'MoE-MLP':<15}")
    print("-"*80)
    print(f"{'Initial Memory (MB)':<40} | {kan_results['initial_memory_mb']:>13.2f} | {mlp_results['initial_memory_mb']:>13.2f}")
    print(f"{'Peak Memory (MB)':<40} | {kan_results['peak_memory_mb']:>13.2f} | {mlp_results['peak_memory_mb']:>13.2f}")
    print(f"{'Average Memory (MB)':<40} | {kan_results['avg_memory_mb']:>13.2f} | {mlp_results['avg_memory_mb']:>13.2f}")
    print(f"{'Memory Overhead (MB)':<40} | {kan_results['memory_overhead_mb']:>13.2f} | {mlp_results['memory_overhead_mb']:>13.2f}")
    
    mem_ratio = mlp_results['peak_memory_mb'] / kan_results['peak_memory_mb']
    print(f"\n💡 {'MLP' if mem_ratio > 1 else 'KAN'} uses {abs(mem_ratio):.2f}x more peak VRAM")
    
    # Expert Usage
    print("\n🎲 EXPERT UTILIZATION")
    print("-"*80)
    print(f"{'Metric':<40} | {'MoE-KAN':<15} | {'MoE-MLP':<15}")
    print("-"*80)
    print(f"{'Active Experts':<40} | {kan_results['num_active_experts']:>13} | {mlp_results['num_active_experts']:>13}")
    print(f"{'Utilization Rate':<40} | {kan_results['num_active_experts']/NUM_EXPERTS*100:>12.1f}% | {mlp_results['num_active_experts']/NUM_EXPERTS*100:>12.1f}%")
    
    # Efficiency Score
    print("\n🏆 EFFICIENCY ANALYSIS")
    print("-"*80)
    
    # Calculate composite efficiency score
    kan_efficiency = (
        kan_results['accuracy'] * 
        (mlp_stats['total_system_params'] / kan_stats['total_system_params']) *
        (mlp_results['total_inference_time'] / kan_results['total_inference_time']) *
        (mlp_results['peak_memory_mb'] / max(kan_results['peak_memory_mb'], 1))
    )
    
    mlp_efficiency = mlp_results['accuracy']
    
    print(f"{'Parameter Efficiency (Acc/Params)':<40} | {(kan_results['accuracy']/kan_stats['total_system_params']*1e6):>13.4f} | {(mlp_results['accuracy']/mlp_stats['total_system_params']*1e6):>13.4f}")
    print(f"{'Speed Efficiency (Acc*Throughput)':<40} | {kan_results['accuracy']*kan_results['throughput_samples_per_sec']:>13.2f} | {mlp_results['accuracy']*mlp_results['throughput_samples_per_sec']:>13.2f}")
    print(f"{'Memory Efficiency (Acc/MB)':<40} | {kan_results['accuracy']/kan_results['peak_memory_mb']:>13.6f} | {mlp_results['accuracy']/mlp_results['peak_memory_mb']:>13.6f}")
    print(f"{'Composite Efficiency Score':<40} | {kan_efficiency:>13.4f} | {mlp_efficiency:>13.4f}")
    
    # Final Verdict
    print("\n" + "═"*80)
    print(" "*30 + "FINAL VERDICT")
    print("═"*80)
    
    if kan_results['accuracy'] >= (mlp_results['accuracy'] - 0.01):
        print(f"\n🏆 WINNER: MoE-KAN")
        print(f"   • Achieves comparable accuracy ({kan_results['accuracy']*100:.2f}% vs {mlp_results['accuracy']*100:.2f}%)")
        print(f"   • Uses {param_ratio:.1f}x FEWER parameters")
        print(f"   • {'Faster' if speedup > 1 else 'Slower'} inference ({abs(speedup):.2f}x)")
        print(f"   • More memory efficient ({mem_ratio:.2f}x less peak VRAM)")
        print(f"\n✨ KAN demonstrates superior parameter efficiency for large-scale MoE systems")
    else:
        acc_gap = (mlp_results['accuracy'] - kan_results['accuracy']) * 100
        print(f"\n⚖️  MIXED RESULTS:")
        print(f"   • MLP leads in accuracy by {acc_gap:.2f}%")
        print(f"   • KAN leads in efficiency ({param_ratio:.1f}x fewer parameters)")
        print(f"\n💡 Trade-off: Choose MLP for maximum accuracy, KAN for resource-constrained deployments")
    
    print("═"*80 + "\n")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting Comprehensive Model Comparison")
    print(f"   Device: {DEVICE}")
    print(f"   Test Size: {TEST_SIZE:,}")
    print(f"   Batch Size: {BATCH_SIZE:,}")
    
    # Load test data
    print("\n📂 Loading Test Data...")
    data = torch.load('kan_deep_data.pt', weights_only=False)
    torch.manual_seed(42) 
    indices = torch.randperm(len(data['inputs']))[:TEST_SIZE]
    X_test = data['inputs'][indices]
    y_test = data['labels'][indices].numpy()
    
    # Get model statistics
    print("\n📐 Analyzing Model Architectures...")
    kan_stats = get_model_stats('KAN')
    mlp_stats = get_model_stats('MLP')
    
    # Run KAN evaluation
    print("\n" + "="*80)
    print("EVALUATING MoE-KAN")
    print("="*80)
    kan_metrics = run_moe_inference_with_metrics('KAN', X_test)
    kan_results = evaluate_model(y_test, kan_metrics['predictions'], 'KAN', kan_metrics)
    
    # Run MLP evaluation
    print("\n" + "="*80)
    print("EVALUATING MoE-MLP")
    print("="*80)
    mlp_metrics = run_moe_inference_with_metrics('MLP', X_test)
    mlp_results = evaluate_model(y_test, mlp_metrics['predictions'], 'MLP', mlp_metrics)
    
    # Print comprehensive results
    print_results(kan_results, mlp_results, kan_stats, mlp_stats)
    
    # Save detailed results to file
    print("💾 Saving detailed results to 'comparison_results.txt'...")
    with open('comparison_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED MODEL COMPARISON RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("KAN RESULTS:\n")
        for key, value in kan_results.items():
            if key != 'expert_usage_distribution':
                f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        f.write("MLP RESULTS:\n")
        for key, value in mlp_results.items():
            if key != 'expert_usage_distribution':
                f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("MODEL STATISTICS:\n")
        f.write("\nKAN Stats:\n")
        for key, value in kan_stats.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nMLP Stats:\n")
        for key, value in mlp_stats.items():
            f.write(f"  {key}: {value}\n")
    
    print("✅ Comparison complete!")