import torch
import pandas as pd
import numpy as np
from collections import Counter

# --- CONFIG ---
DATA_FILE = 'kan_deep_data.pt'
ROUTER_DATA = 'router_train_data.pt'

def get_icd_chapter(code):
    """Maps an ICD-10 code to its primary category (Chapter)"""
    if not code or not isinstance(code, str): return "Unknown"
    first_char = code[0].upper()
    
    # ICD-10 Chapter Mapping (Simplified)
    mapping = {
        'A': 'Infectious', 'B': 'Infectious',
        'C': 'Neoplasms', 'D': 'Neoplasms/Blood',
        'E': 'Endocrine',
        'F': 'Mental/Behavioral',
        'G': 'Nervous System',
        'H': 'Eye/Ear',
        'I': 'Circulatory (Heart)',
        'J': 'Respiratory',
        'K': 'Digestive',
        'L': 'Skin',
        'M': 'Musculoskeletal',
        'N': 'Genitourinary',
        'O': 'Pregnancy/Childbirth',
        'P': 'Perinatal',
        'Q': 'Congenital',
        'R': 'Abnormal Clinical Findings',
        'S': 'Injury/Poisoning', 'T': 'Injury/Poisoning',
        'V': 'External Causes', 'W': 'External Causes', 'X': 'External Causes', 'Y': 'External Causes',
        'Z': 'Health Status Factors'
    }
    return mapping.get(first_char, "Other")

print("Loading data for clinical analysis...")
data = torch.load(DATA_FILE, weights_only=False)
router_data = torch.load(ROUTER_DATA, weights_only=False)

# 1. Prepare Dataframe
df_diag = pd.DataFrame({
    'label_idx': data['labels'].numpy(),
    'expert_id': router_data['targets'].numpy()
})

# Map index back to actual ICD string
class_map = data['classes']
df_diag['icd_code'] = df_diag['label_idx'].apply(lambda x: class_map[x])
df_diag['chapter'] = df_diag['icd_code'].apply(get_icd_chapter)

# 2. Analyze Expert Specialization
print("\n" + "="*50)
print("EXPERT SPECIALIZATION REPORT")
print("="*50)

expert_specialties = []

for i in range(32):
    expert_mask = df_diag['expert_id'] == i
    expert_data = df_diag[expert_mask]
    
    if len(expert_data) == 0: continue
    
    # Calculate top chapters for this expert
    chapter_counts = expert_data['chapter'].value_counts()
    top_chapter = chapter_counts.index[0]
    purity = (chapter_counts.iloc[0] / len(expert_data)) * 100
    
    expert_specialties.append({
        'Expert': i,
        'Primary Domain': top_chapter,
        'Domain Purity %': round(purity, 2),
        'Sample Count': len(expert_data),
        'Unique Codes': expert_data['icd_code'].nunique()
    })

report_df = pd.DataFrame(expert_specialties)

# Compute global purity
avg_purity = report_df['Domain Purity %'].mean()

# Output file path
output_file = "expert_domain_purity_report.txt"

# Write to TXT file
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Expert Specialties Report\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(report_df.to_string(index=False))
    
    f.write("\n\n" + "=" * 60 + "\n")
    f.write(f"Overall Semantic Clustering Purity: {avg_purity:.2f}%\n")

print(f"Report successfully saved to: {output_file}")