import pandas as pd
import json
import os
from tqdm import tqdm

# Paths
BASE_DIR = "/Users/karanvirkhanna/BigCodeNet"
PAIRS_CSV = os.path.join(BASE_DIR, "pairs.csv")
OUTPUT_FILE = "bigcodenet_pairs.json"

def prepare_data():
    print(f"Reading {PAIRS_CSV}...")
    try:
        df = pd.read_csv(PAIRS_CSV)
    except FileNotFoundError:
        print(f"Error: {PAIRS_CSV} not found.")
        return

    print(f"Found {len(df)} pairs.")
    
    data = []
    
    # Iterate over rows
    print("Reading code files...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        py_path = os.path.join(BASE_DIR, row['py_relative_path'])
        java_path = os.path.join(BASE_DIR, row['java_relative_path'])
        
        try:
            with open(py_path, 'r', encoding='utf-8') as f:
                python_code = f.read()
            
            with open(java_path, 'r', encoding='utf-8') as f:
                java_code = f.read()
                
            data.append({
                'python': python_code,
                'java': java_code
            })
        except Exception as e:
            # print(f"Error reading {py_path} or {java_path}: {e}")
            continue

    print(f"Successfully loaded {len(data)} pairs.")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_data()
