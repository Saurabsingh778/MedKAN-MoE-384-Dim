"""
MAXIMUM PERFORMANCE VERSION - RAW 384-DIM
- Removed PCA and Scaling to keep actual noise/signal
- Keeps full 384 dimensions from SentenceTransformer
- Vectorized operations and memory-efficient saving
"""

import pandas as pd
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import re

# --- CONFIG ---
DATA_DIR = "./titans_medical_data_v3_clean"
FILES = {
    "direct": "titans_1_direct_attributes.jsonl",
    "noisy": "titans_2_noisy_robustness.jsonl",
    "needle": "titans_3_needle_reasoning.jsonl"
}

EMBEDDING_BATCH_SIZE = 1024

# Pre-compile regex for faster label cleaning
CODE_PATTERN = re.compile(r'(?:Code:\s*)?([A-Z]\d+\.?\d*)')

def clean_label_fast(text):
    """Optimized label cleaning with regex"""
    if not isinstance(text, str) or not text:
        return ""
    match = CODE_PATTERN.search(text)
    return match.group(1) if match else text.split()[0].split('(')[0].strip()

def load_and_balance_optimized():
    """PHASE 1-3: Load and balance data"""
    print("=" * 70)
    print("PHASE 1: DATA LOADING")
    print("=" * 70)
    
    all_data = {}
    total_rows = 0
    
    for source, f_name in FILES.items():
        path = os.path.join(DATA_DIR, f_name)
        if not os.path.exists(path):
            print(f"❌ Error: {f_name} not found!")
            return None
        
        print(f"Loading {source}...", end=" ")
        df = pd.read_json(path, lines=True, dtype={'output': str, 'input': str})
        df['clean_label'] = df['output'].apply(clean_label_fast)
        mask = df['clean_label'].str.len() > 0
        df = df[mask].copy()
        df['source'] = source
        all_data[source] = df
        total_rows += len(df)
        print(f"✓ {len(df):,} rows")
    
    print("\nPHASE 2: BUILDING INDEX & FILTERING")
    code_index = {}
    for source, df in all_data.items():
        for code, group_df in df.groupby('clean_label'):
            if code not in code_index:
                code_index[code] = {}
            code_index[code][source] = group_df.index.tolist()
    
    valid_codes = [c for c, s in code_index.items() if len(s) == 3]
    print(f"Valid codes (in all 3 files): {len(valid_codes):,}")

    print("\nPHASE 3: BALANCED SAMPLING")
    all_indices_by_source = {source: [] for source in FILES.keys()}
    for code in tqdm(valid_codes, desc="Sampling", unit="code"):
        min_count = min(len(code_index[code][s]) for s in FILES.keys())
        for source in FILES.keys():
            indices = np.array(code_index[code][source])
            sampled = np.random.choice(indices, size=min_count, replace=False)
            all_indices_by_source[source].extend(sampled.tolist())

    balanced_dfs = [all_data[source].loc[all_indices_by_source[source]] for source in FILES.keys()]
    final_df = pd.concat(balanced_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return final_df

def process_embeddings_raw(df):
    """PHASE 4: Generate Raw 384-dim Embeddings"""
    print("\n" + "=" * 70)
    print("PHASE 4: RAW EMBEDDING GENERATION")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Device: {device} | Batch Size: {EMBEDDING_BATCH_SIZE}")
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    model.eval() 
    
    texts = df['input'].tolist()
    print(f"✓ Encoding {len(texts):,} texts to 384-dim...")
    
    with torch.no_grad():
        embeddings = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False # Keeping raw signal
        )
    
    print(f"✓ Final embedding shape: {embeddings.shape}")
    return embeddings

def save_dataset(X, y, classes, df):
    """PHASE 6: Saving Raw Data"""
    print("\n" + "=" * 70)
    print("PHASE 6: SAVING")
    print("=" * 70)
    
    output_file = 'kan_deep_data.pt'
    
    save_dict = {
        'inputs': torch.tensor(X, dtype=torch.float32),
        'labels': torch.tensor(y, dtype=torch.long),
        'classes': classes,
        'metadata': {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(classes),
            'sources': df['source'].value_counts().to_dict(),
            'pca_applied': False,
            'variance_retained': 1.0, # Full signal kept
            'model_source': 'all-MiniLM-L6-v2'
        }
    }
    
    print(f"Saving to {output_file}...", end=" ")
    torch.save(save_dict, output_file)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ {file_size_mb:.2f} MB")
    
    return output_file

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # 1-3: Load
    df = load_and_balance_optimized()
    
    # 4: Raw Embeddings
    X = process_embeddings_raw(df)
    
    # Label Encoding
    le = LabelEncoder()
    y = le.fit_transform(df['clean_label'])
    
    # 6: Save
    save_dataset(X, y, le.classes_, df)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Processed {len(X):,} samples in {int(elapsed//60)}m {int(elapsed%60)}s")
    print("🎉 All done! No PCA applied, actual 384 dims retained.")