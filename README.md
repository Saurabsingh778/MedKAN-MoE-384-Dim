# MedKAN-MoE  
**High-Efficiency Mixture of Experts for ICD-10 Coding**  
*Benchmarking Kolmogorov-Arnold Networks vs. MLPs in Extreme Multi-Label Medical Classification*

---

## 📌 Overview

**MedKAN-MoE** is a research-oriented repository dedicated to evaluating **parameter-efficient neural architectures** for large-scale medical coding tasks.  
The project performs a rigorous head-to-head comparison between **Kolmogorov-Arnold Networks (KAN)** and traditional **Multilayer Perceptrons (MLP)** within a **Mixture of Experts (MoE)** framework.

The target task is extreme multi-label classification: mapping clinical concepts (analogous to SNOMED CT-style descriptions) to **ICD-10-CM codes**, spanning **19,756 unique medical labels**.

The core research question addressed is:

> *How much “intelligence per parameter” can KAN-based experts deliver compared to conventional MLPs under identical routing and data conditions?*

---

## 🏥 The Clinical Challenge

Automated ICD-10-CM coding is a cornerstone problem in medical informatics, yet it poses several major challenges:

- **Extreme Label Cardinality**  
  Over 19,000 output classes with a heavy long-tail distribution of rare diseases.

- **High Semantic Granularity**  
  Subtle textual variations (e.g., fracture location or laterality) map to distinct billing codes.

- **Deployment Constraints**  
  Real-time hospital systems require low-latency inference and minimal memory footprints.

Conventional dense neural networks often scale poorly in this regime.  
**MedKAN-MoE** addresses this by semantically partitioning the problem space and assigning **specialized, compact expert networks** to each clinical sub-domain.

---

## 🔬 Methodology: KAN vs. MLP Experts

Clinical text is first embedded into a **384-dimensional semantic space** using the `all-MiniLM-L6-v2` sentence transformer.  
A learned router dynamically dispatches each embedding to one of **32 semantic experts**.

Both expert types operate under **identical MoE routing and training conditions**.

### 1. MoE-MLP Baseline (Traditional)

- **Architecture**: Deep linear layers with ReLU activations and Batch Normalization  
- **Training Philosophy**: Parameter-heavy memorization  
- **Scale**: ~5.6 million parameters per expert  

### 2. MoE-KAN (Proposed)

- **Architecture**: Kolmogorov-Arnold Networks with learnable B-spline activations on every edge  
- **Training Philosophy**: Function-level expressivity with minimal parameters  
- **Scale**: ~410k parameters per expert (**13.8× smaller**)  

---

## 📊 Experimental Results & Analysis

Benchmarks were conducted on **10,000 randomized clinical samples** using identical routing, batching, and evaluation protocols.

### Summary of Key Findings

| Evaluation Metric | MoE-KAN (Proposed) | MoE-MLP (Baseline) | Research Insight |
|------------------|------------------|------------------|----------------|
| Expert Parameters | ~410k | ~5.6M | KAN is **13.8× more parameter-efficient** |
| Total System Params | 119.3M | 181.5M | 34% smaller system footprint |
| Accuracy | 96.73% | 98.13% | MLP retains a slight +1.4% edge |
| Weighted F1-Score | 0.9663 | 0.9803 | Strong rare-class performance for both |
| Inference Throughput | 2,385 samples/sec | 2,087 samples/sec | KAN is **14.3% faster** |
| Peak VRAM Usage | 37.53 MB | 44.43 MB | 15.5% lower memory usage |
| Composite Efficiency | **1.000 (Winner)** | 0.804 | Best accuracy–efficiency trade-off |

---

## 🧠 Research Conclusion

While the MoE-MLP architecture achieves marginally higher raw accuracy, this gain comes at a **dramatically higher parameter and memory cost**.

**MoE-KAN delivers comparable clinical precision while being:**

- 13.8× smaller per expert  
- Faster at inference time  
- Significantly more memory-efficient  

These results position **MedKAN-MoE** as a strong blueprint for **deployable, resource-efficient medical AI systems**, especially in real-world hospital and edge environments.

---
