import pandas as pd
from transformers import PLBartForConditionalGeneration, PLBartTokenizer
import os
from tqdm import tqdm

# Configuration
MODEL_NAME = "uclanlp/plbart-base"
DATA_DIR = "training_pairs"
PAIRS_FILE = os.path.join(DATA_DIR, "pairs.csv")
OUTPUT_DIR = "translated_output"
NUM_SAMPLES = 5  # Number of samples to translate for verification

def load_model():
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = PLBartTokenizer.from_pretrained(MODEL_NAME)
    print(f"Available language codes: {tokenizer.lang_code_to_id}")
    model = PLBartForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model

def translate_code(code, tokenizer, model):
    # Find the correct language code for Java
    java_lang_id = tokenizer.lang_code_to_id.get("java")
    if java_lang_id is None:
        # Try to find a key that looks like java
        for key in tokenizer.lang_code_to_id:
            if "java" in key.lower():
                java_lang_id = tokenizer.lang_code_to_id[key]
                print(f"Using language code: {key} (ID: {java_lang_id})")
                break
    
    if java_lang_id is None:
        raise ValueError("Could not find Java language code in tokenizer.")

    inputs = tokenizer(code, return_tensors="pt", max_length=512, truncation=True)
    translated_tokens = model.generate(**inputs, decoder_start_token_id=java_lang_id)
    translated_code = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_code

def translate_sample(index, tokenizer, model, df):
    row = df.iloc[index]
    py_path = os.path.join(DATA_DIR, row['py_relative_path'])
    java_path = os.path.join(DATA_DIR, row['java_relative_path'])
    
    with open(py_path, 'r', encoding='utf-8') as f:
        python_code = f.read()
    
    with open(java_path, 'r', encoding='utf-8') as f:
        java_target_code = f.read()

    translated_java = translate_code(python_code, tokenizer, model)
    
    return python_code, java_target_code, translated_java

def main():
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load data
    print(f"Loading data from {PAIRS_FILE}...")
    df = pd.read_csv(PAIRS_FILE)
    
    # Load model
    tokenizer, model = load_model()

    # Process samples
    print(f"Translating first {NUM_SAMPLES} samples...")
    for i in range(NUM_SAMPLES):
        try:
            print(f"\n--- Sample {i+1} ---")
            python_code, java_target_code, translated_java = translate_sample(i, tokenizer, model, df)
            
            # Save output
            output_filename = f"sample_{i+1}_translated.java"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_java)
            
            print(f"Translated code saved to {output_path}")
            print("Original Java (First 100 chars):")
            print(java_target_code[:100] + "...")
            print("Translated Java (First 100 chars):")
            print(translated_java[:100] + "...")

        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")

if __name__ == "__main__":
    main()
