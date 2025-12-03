import pandas as pd
import evaluate
import os
from tqdm import tqdm

# Configuration
DATA_DIR = "training_pairs"
PAIRS_FILE = os.path.join(DATA_DIR, "pairs.csv")
OUTPUT_DIR = "translated_output"
NUM_SAMPLES = 5

def evaluate_translations(num_samples=NUM_SAMPLES):
    # Load metric
    bleu = evaluate.load("sacrebleu")

    # Load data
    print(f"Loading data from {PAIRS_FILE}...")
    df = pd.read_csv(PAIRS_FILE)
    
    predictions = []
    references = []

    print(f"Evaluating first {num_samples} samples...")
    for i in range(num_samples):
        row = df.iloc[i]
        java_path = os.path.join(DATA_DIR, row['java_relative_path'])
        
        # Load ground truth
        try:
            with open(java_path, 'r', encoding='utf-8') as f:
                java_target_code = f.read()
            references.append([java_target_code])
        except Exception as e:
            print(f"Error reading reference {i+1}: {e}")
            continue

        # Load prediction
        output_filename = f"sample_{i+1}_translated.java"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                translated_code = f.read()
            predictions.append(translated_code)
        except Exception as e:
            print(f"Error reading prediction {i+1}: {e}")
            # If prediction missing, append empty string to keep alignment? 
            # Or skip. For now, let's assume we only evaluate what we generated.
            # But references was already appended. So we must append something.
            predictions.append("")

            # But references was already appended. So we must append something.
            predictions.append("")

    # Compute metrics
    if predictions:
        results = bleu.compute(predictions=predictions, references=references)
        return results
    else:
        return None

def main():
    results = evaluate_translations(NUM_SAMPLES)
    if results:
        print("\n--- Evaluation Results ---")
        print(f"BLEU Score: {results['score']}")
        print(f"Detailed results: {results}")
    else:
        print("No predictions to evaluate.")

if __name__ == "__main__":
    main()
