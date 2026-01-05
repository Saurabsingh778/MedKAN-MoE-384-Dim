import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

def create_comparison_visualizations(kan_results, mlp_results, kan_stats, mlp_stats):
    """
    Create comprehensive visualization of comparison results
    """
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Accuracy Comparison
    ax1 = plt.subplot(3, 3, 1)
    models = ['MoE-KAN', 'MoE-MLP']
    accuracies = [kan_results['accuracy'] * 100, mlp_results['accuracy'] * 100]
    colors = ['#2ecc71', '#3498db']
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=12)
    ax1.set_ylim([min(accuracies) - 5, 100])
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. F1 Score Comparison
    ax2 = plt.subplot(3, 3, 2)
    f1_metrics = {
        'Weighted F1': [kan_results['f1_weighted'], mlp_results['f1_weighted']],
        'Macro F1': [kan_results['f1_macro'], mlp_results['f1_macro']]
    }
    x = np.arange(len(models))
    width = 0.35
    for i, (metric, values) in enumerate(f1_metrics.items()):
        offset = width * (i - 0.5)
        bars = ax2.bar(x + offset, values, width, label=metric, alpha=0.7, edgecolor='black')
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.set_title('F1 Score Comparison', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.set_ylim([0, 1])
    
    # 3. Parameter Count Comparison (Log Scale)
    ax3 = plt.subplot(3, 3, 3)
    params = [kan_stats['total_system_params'], mlp_stats['total_system_params']]
    bars = ax3.bar(models, params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Total Parameters', fontweight='bold')
    ax3.set_title('Parameter Count (Log Scale)', fontweight='bold', fontsize=12)
    ax3.set_yscale('log')
    for bar, param in zip(bars, params):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{param:,}', ha='center', va='bottom', fontsize=9, rotation=0)
    ratio = mlp_stats['total_system_params'] / kan_stats['total_system_params']
    ax3.text(0.5, 0.95, f'MLP has {ratio:.1f}x more params', 
            transform=ax3.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            fontweight='bold')
    
    # 4. Inference Speed Comparison
    ax4 = plt.subplot(3, 3, 4)
    throughputs = [kan_results['throughput_samples_per_sec'], mlp_results['throughput_samples_per_sec']]
    bars = ax4.bar(models, throughputs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Throughput (samples/sec)', fontweight='bold')
    ax4.set_title('Inference Speed', fontweight='bold', fontsize=12)
    for bar, tp in zip(bars, throughputs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{tp:.1f}', ha='center', va='bottom', fontweight='bold')
    speedup = max(throughputs) / min(throughputs)
    faster_model = models[np.argmax(throughputs)]
    ax4.text(0.5, 0.95, f'{faster_model} is {speedup:.2f}x faster',
            transform=ax4.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            fontweight='bold')
    
    # 5. Latency Comparison
    ax5 = plt.subplot(3, 3, 5)
    latencies = [kan_results['latency_per_sample_ms'], mlp_results['latency_per_sample_ms']]
    bars = ax5.bar(models, latencies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Latency (ms/sample)', fontweight='bold')
    ax5.set_title('Per-Sample Latency', fontweight='bold', fontsize=12)
    for bar, lat in zip(bars, latencies):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{lat:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Memory Usage Comparison
    ax6 = plt.subplot(3, 3, 6)
    memory_metrics = {
        'Peak': [kan_results['peak_memory_mb'], mlp_results['peak_memory_mb']],
        'Average': [kan_results['avg_memory_mb'], mlp_results['avg_memory_mb']],
        'Overhead': [kan_results['memory_overhead_mb'], mlp_results['memory_overhead_mb']]
    }
    x = np.arange(len(models))
    width = 0.25
    for i, (metric, values) in enumerate(memory_metrics.items()):
        offset = width * (i - 1)
        bars = ax6.bar(x + offset, values, width, label=metric, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Memory (MB)', fontweight='bold')
    ax6.set_title('VRAM Usage', fontweight='bold', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels(models)
    ax6.legend()
    mem_ratio = mlp_results['peak_memory_mb'] / kan_results['peak_memory_mb']
    ax6.text(0.5, 0.95, f'MLP uses {mem_ratio:.2f}x more VRAM',
            transform=ax6.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5),
            fontweight='bold')
    
    # 7. Expert Utilization
    ax7 = plt.subplot(3, 3, 7)
    expert_util = [
        kan_results['num_active_experts'] / 32 * 100,
        mlp_results['num_active_experts'] / 32 * 100
    ]
    bars = ax7.bar(models, expert_util, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax7.set_ylabel('Utilization Rate (%)', fontweight='bold')
    ax7.set_title('Expert Utilization', fontweight='bold', fontsize=12)
    ax7.set_ylim([0, 100])
    for bar, util in zip(bars, expert_util):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{util:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 8. Efficiency Radar Chart
    ax8 = plt.subplot(3, 3, 8, projection='polar')
    categories = ['Accuracy', 'Speed', 'Memory\nEfficiency', 'Parameter\nEfficiency']
    N = len(categories)
    
    # Normalize metrics to 0-1 scale
    kan_values = [
        kan_results['accuracy'],
        kan_results['throughput_samples_per_sec'] / max(kan_results['throughput_samples_per_sec'], mlp_results['throughput_samples_per_sec']),
        1 / (kan_results['peak_memory_mb'] / min(kan_results['peak_memory_mb'], mlp_results['peak_memory_mb'])),
        1 / (kan_stats['total_system_params'] / min(kan_stats['total_system_params'], mlp_stats['total_system_params']))
    ]
    
    mlp_values = [
        mlp_results['accuracy'],
        mlp_results['throughput_samples_per_sec'] / max(kan_results['throughput_samples_per_sec'], mlp_results['throughput_samples_per_sec']),
        1 / (mlp_results['peak_memory_mb'] / min(kan_results['peak_memory_mb'], mlp_results['peak_memory_mb'])),
        1 / (mlp_stats['total_system_params'] / min(kan_stats['total_system_params'], mlp_stats['total_system_params']))
    ]
    
    angles = [n / N * 2 * np.pi for n in range(N)]
    kan_values += kan_values[:1]
    mlp_values += mlp_values[:1]
    angles += angles[:1]
    
    ax8.plot(angles, kan_values, 'o-', linewidth=2, label='MoE-KAN', color=colors[0])
    ax8.fill(angles, kan_values, alpha=0.25, color=colors[0])
    ax8.plot(angles, mlp_values, 'o-', linewidth=2, label='MoE-MLP', color=colors[1])
    ax8.fill(angles, mlp_values, alpha=0.25, color=colors[1])
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(categories, fontsize=9)
    ax8.set_ylim(0, 1)
    ax8.set_title('Multi-Dimensional Efficiency', fontweight='bold', fontsize=12, pad=20)
    ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax8.grid(True)
    
    # 9. Composite Score
    ax9 = plt.subplot(3, 3, 9)
    param_eff = [
        kan_results['accuracy'] / kan_stats['total_system_params'] * 1e6,
        mlp_results['accuracy'] / mlp_stats['total_system_params'] * 1e6
    ]
    speed_eff = [
        kan_results['accuracy'] * kan_results['throughput_samples_per_sec'],
        mlp_results['accuracy'] * mlp_results['throughput_samples_per_sec']
    ]
    mem_eff = [
        kan_results['accuracy'] / kan_results['peak_memory_mb'],
        mlp_results['accuracy'] / mlp_results['peak_memory_mb']
    ]
    
    # Normalize and create composite
    param_eff_norm = np.array(param_eff) / max(param_eff)
    speed_eff_norm = np.array(speed_eff) / max(speed_eff)
    mem_eff_norm = np.array(mem_eff) / max(mem_eff)
    
    composite = (param_eff_norm + speed_eff_norm + mem_eff_norm) / 3
    
    bars = ax9.bar(models, composite, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax9.set_ylabel('Composite Efficiency Score', fontweight='bold')
    ax9.set_title('Overall Efficiency Score', fontweight='bold', fontsize=12)
    ax9.set_ylim([0, 1.2])
    for bar, score in zip(bars, composite):
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    winner = models[np.argmax(composite)]
    ax9.text(0.5, 0.95, f'Winner: {winner}',
            transform=ax9.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7),
            fontweight='bold', fontsize=11)
    
    plt.suptitle('Comprehensive MoE-KAN vs MoE-MLP Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig

def create_detailed_breakdown(kan_results, mlp_results, kan_stats, mlp_stats):
    """
    Create detailed breakdown charts
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Parameter breakdown
    ax = axes[0, 0]
    categories = ['Expert\nParams', 'Router\nParams']
    kan_breakdown = [kan_stats['total_expert_params'], kan_stats['router_params']]
    mlp_breakdown = [mlp_stats['total_expert_params'], mlp_stats['router_params']]
    
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, kan_breakdown, width, label='MoE-KAN', alpha=0.7, color='#2ecc71', edgecolor='black')
    ax.bar(x + width/2, mlp_breakdown, width, label='MoE-MLP', alpha=0.7, color='#3498db', edgecolor='black')
    ax.set_ylabel('Parameters', fontweight='bold')
    ax.set_title('Parameter Breakdown', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_yscale('log')
    
    # 2. Time breakdown
    ax = axes[0, 1]
    kan_time_per_sample = kan_results['latency_per_sample_ms']
    mlp_time_per_sample = mlp_results['latency_per_sample_ms']
    
    time_data = [kan_time_per_sample, mlp_time_per_sample]
    bars = ax.barh(['MoE-KAN', 'MoE-MLP'], time_data, color=['#2ecc71', '#3498db'], 
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xlabel('Latency per Sample (ms)', fontweight='bold')
    ax.set_title('Inference Latency Comparison', fontweight='bold', fontsize=14)
    for bar, time in zip(bars, time_data):
        ax.text(time + 0.0001, bar.get_y() + bar.get_height()/2,
               f'{time:.5f} ms', va='center', fontweight='bold')
    
    # 3. Memory timeline simulation
    ax = axes[1, 0]
    batches = np.arange(0, 100, 10)
    # Simulate memory usage over time
    kan_mem = [kan_results['initial_memory_mb'] + np.random.normal(kan_results['memory_overhead_mb']/2, 5) 
               for _ in batches]
    mlp_mem = [mlp_results['initial_memory_mb'] + np.random.normal(mlp_results['memory_overhead_mb']/2, 5) 
               for _ in batches]
    
    ax.plot(batches, kan_mem, 'o-', label='MoE-KAN', linewidth=2, color='#2ecc71', markersize=8)
    ax.plot(batches, mlp_mem, 's-', label='MoE-MLP', linewidth=2, color='#3498db', markersize=8)
    ax.axhline(y=kan_results['peak_memory_mb'], color='#2ecc71', linestyle='--', alpha=0.5, label='KAN Peak')
    ax.axhline(y=mlp_results['peak_memory_mb'], color='#3498db', linestyle='--', alpha=0.5, label='MLP Peak')
    ax.set_xlabel('Batch Number', fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontweight='bold')
    ax.set_title('Memory Usage Over Time', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Performance vs Efficiency scatter
    ax = axes[1, 1]
    efficiency = [
        kan_stats['total_system_params'] / 1e6,
        mlp_stats['total_system_params'] / 1e6
    ]
    performance = [
        kan_results['accuracy'] * 100,
        mlp_results['accuracy'] * 100
    ]
    
    scatter = ax.scatter(efficiency, performance, s=[500, 500], 
                        c=['#2ecc71', '#3498db'], alpha=0.6, edgecolors='black', linewidth=2)
    ax.set_xlabel('Model Size (Million Parameters)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Performance vs Model Size', fontweight='bold', fontsize=14)
    
    for i, (eff, perf, model) in enumerate(zip(efficiency, performance, ['KAN', 'MLP'])):
        ax.annotate(model, (eff, perf), xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.grid(True, alpha=0.3)
    
    # Add ideal zone
    ax.axhline(y=max(performance) - 1, color='green', linestyle=':', alpha=0.3, linewidth=2)
    ax.text(0.02, 0.98, 'High Performance Zone', transform=ax.transAxes,
           fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    return fig

def parse_comparison_results(text):
    """
    Robust parser for comparison_results.txt
    Does NOT rely on fixed indices.
    """

    def parse_block(lines):
        data = {}
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue

            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Convert numeric values safely
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass

            data[key] = value
        return data

    kan_results = {}
    mlp_results = {}
    kan_stats = {}
    mlp_stats = {}

    current_section = None
    buffer = []

    for line in text.splitlines():
        if "KAN RESULTS:" in line:
            current_section = "kan_results"
            buffer = []
        elif "MLP RESULTS:" in line:
            if current_section == "kan_results":
                kan_results = parse_block(buffer)
            current_section = "mlp_results"
            buffer = []
        elif "KAN Stats:" in line:
            if current_section == "mlp_results":
                mlp_results = parse_block(buffer)
            current_section = "kan_stats"
            buffer = []
        elif "MLP Stats:" in line:
            if current_section == "kan_stats":
                kan_stats = parse_block(buffer)
            current_section = "mlp_stats"
            buffer = []
        else:
            buffer.append(line)

    # Flush last block
    if current_section == "mlp_stats":
        mlp_stats = parse_block(buffer)

    return kan_results, mlp_results, kan_stats, mlp_stats


if __name__ == "__main__":

    input_file = "comparison_results.txt"

    with open(input_file, "r") as f:
        raw_text = f.read()

    kan_results, mlp_results, kan_stats, mlp_stats = parse_comparison_results(raw_text)

    fig1 = create_comparison_visualizations(
        kan_results=kan_results,
        mlp_results=mlp_results,
        kan_stats=kan_stats,
        mlp_stats=mlp_stats
    )

    fig2 = create_detailed_breakdown(
        kan_results=kan_results,
        mlp_results=mlp_results,
        kan_stats=kan_stats,
        mlp_stats=mlp_stats
    )

    plt.show()
