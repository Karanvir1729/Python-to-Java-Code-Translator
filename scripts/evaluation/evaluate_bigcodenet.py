import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import pandas as pd
import evaluate
from tqdm import tqdm
import os
import subprocess
import tempfile
from codebleu import calc_codebleu
import random

# Configuration
MODEL_PATH = "./codet5_model/codet5_fine_tuned"
BIGCODENET_PATH = "/Users/karanvirkhanna/BigCodeNet"
PAIRS_FILE = os.path.join(BIGCODENET_PATH, "pairs.csv")
OUTPUT_FILE = "bigcodenet_evaluation_results.csv"
BASE_MODEL = "Salesforce/codet5-small"
NUM_SAMPLES = 1050 # Slightly more than 1000 to be safe

def load_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading fine-tuned model from {MODEL_PATH}...")
        model_name = MODEL_PATH
    else:
        print(f"Fine-tuned model not found at {MODEL_PATH}")
        print(f"Loading base model {BASE_MODEL} for demonstration...")
        model_name = BASE_MODEL

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    model.to(device)
    return tokenizer, model, device

def load_bigcodenet_data(pairs_file, num_samples):
    print(f"Loading pairs from {pairs_file}...")
    df = pd.read_csv(pairs_file)
    
    # Sample data
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42)
    
    data = []
    print(f"Reading {len(df)} file pairs...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        py_path = os.path.join(BIGCODENET_PATH, row['py_relative_path'])
        java_path = os.path.join(BIGCODENET_PATH, row['java_relative_path'])
        
        try:
            with open(py_path, 'r') as f:
                python_code = f.read()
            with open(java_path, 'r') as f:
                java_code = f.read()
                
            data.append({
                "python": python_code,
                "java": java_code,
                "id": row['problem_id']
            })
        except Exception as e:
            # print(f"Error reading pair: {e}")
            continue
            
    return data

def check_compilation(java_code):
    """
    Wraps code in a class and checks if it compiles.
    Returns True if compiles, False otherwise.
    """
    class_name = "Main"
    code_to_compile = java_code
    
    # Simple heuristic to find class name or wrap
    if "public class" in java_code:
        try:
            parts = java_code.split("public class")[1].strip().split()
            class_name = parts[0].split("{")[0].strip()
            # Handle generics or extends
            class_name = class_name.split("<")[0].strip()
        except:
            pass
    elif "class " in java_code:
        try:
            parts = java_code.split("class")[1].strip().split()
            class_name = parts[0].split("{")[0].strip()
            class_name = class_name.split("<")[0].strip()
        except:
            pass
    else:
        # Wrap it
        code_to_compile = f"public class {class_name} {{\n{java_code}\n}}"

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, f"{class_name}.java")
        with open(file_path, "w") as f:
            f.write(code_to_compile)
        
        try:
            subprocess.check_output(["javac", file_path], stderr=subprocess.STDOUT)
            return True
        except subprocess.CalledProcessError:
            return False

def calculate_codebleu(reference, prediction, lang="java"):
    try:
        # Attempting with weights argument as per recent findings
        result = calc_codebleu([reference], [prediction], lang=lang, weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
        return result
    except Exception as e:
        # Fallback or return 0 if it fails
        return {"codebleu": 0.0, "ngram_match_score": 0.0, "weighted_ngram_match_score": 0.0, "syntax_match_score": 0.0, "dataflow_match_score": 0.0}

def generate_and_evaluate(data, tokenizer, model, device):
    results = []
    
    print(f"Evaluating {len(data)} examples...")
    
    for item in tqdm(data):
        python_code = item['python']
        reference_java = item['java']
        
        # Truncate input if too long to avoid errors, though tokenizer handles truncation
        input_text = "Translate Python to Java: " + python_code
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True, 
            padding="max_length"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs, 
                max_length=512, 
                num_beams=1, 
                early_stopping=True
            )
        
        prediction = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        # 1. Compilation
        is_compiled = check_compilation(prediction)
        
        # 2. CodeBLEU
        cb_metrics = calculate_codebleu(reference_java, prediction)
        
        results.append({
            "id": item['id'],
            "python": python_code,
            "reference": reference_java,
            "prediction": prediction,
            "compiled": is_compiled,
            "codebleu": cb_metrics['codebleu'],
            "syntax_match": cb_metrics['syntax_match_score'],
            "dataflow_match": cb_metrics['dataflow_match_score']
        })

        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        
    return results

def main():
    tokenizer, model, device = load_model()
    data = load_bigcodenet_data(PAIRS_FILE, NUM_SAMPLES)
    
    if not data:
        print("No data loaded. Exiting.")
        return

    results = generate_and_evaluate(data, tokenizer, model, device)
    
    # Calculate Aggregates
    df = pd.DataFrame(results)
    
    bleu = evaluate.load("sacrebleu")
    predictions = df['prediction'].tolist()
    references = [[r] for r in df['reference'].tolist()]
    bleu_score = bleu.compute(predictions=predictions, references=references)['score']
    
    avg_codebleu = df['codebleu'].mean()
    avg_syntax = df['syntax_match'].mean()
    avg_dataflow = df['dataflow_match'].mean()
    compilation_rate = df['compiled'].mean()
    
    print("\n--- BigCodeNet Evaluation Results ---")
    print(f"Samples: {len(df)}")
    print(f"BLEU Score: {bleu_score:.2f}")
    print(f"CodeBLEU: {avg_codebleu:.4f}")
    print(f"  - Syntax Match: {avg_syntax:.4f}")
    print(f"  - Dataflow Match: {avg_dataflow:.4f}")
    print(f"Compilation Rate: {compilation_rate:.2%}")
    
    # Save results
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDetailed results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
