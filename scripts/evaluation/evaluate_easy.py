import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import json
import pandas as pd
import evaluate
from tqdm import tqdm
import os
import subprocess
import tempfile
from codebleu import calc_codebleu

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../models/codet5_fine_tuned")
DATA_PATH = os.path.join(BASE_DIR, "../../data/easy_code_pairs.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "../../results/easy_evaluation_results.csv")
BASE_MODEL = "Salesforce/codet5-small"

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

def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def check_compilation(java_code):
    """
    Wraps code in a class and checks if it compiles.
    Returns True if compiles, False otherwise.
    """
    # If the code already contains a class, we try to save it with that name.
    # Since we are using easy tests that likely output "public class Main", we handle that.
    
    class_name = "Main"
    code_to_compile = java_code
    
    if "public class" in java_code:
        try:
            parts = java_code.split("public class")[1].strip().split()
            class_name = parts[0].split("{")[0].strip()
        except:
            pass
    elif "class " in java_code:
        try:
            parts = java_code.split("class")[1].strip().split()
            class_name = parts[0].split("{")[0].strip()
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

def check_execution(java_code):
    # For easy tests, we assume no input is needed or we can't easily provide it.
    # We just check if it runs without error.
    
    class_name = "Main"
    code_to_run = java_code
    
    if "public class" in java_code:
        try:
            parts = java_code.split("public class")[1].strip().split()
            class_name = parts[0].split("{")[0].strip()
        except:
            pass
    elif "class " in java_code:
        try:
            parts = java_code.split("class")[1].strip().split()
            class_name = parts[0].split("{")[0].strip()
        except:
            pass
    else:
        code_to_run = f"public class {class_name} {{\n{java_code}\n}}"

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, f"{class_name}.java")
        with open(file_path, "w") as f:
            f.write(code_to_run)
        
        try:
            # Compile
            subprocess.check_output(["javac", file_path], stderr=subprocess.STDOUT)
            # Run
            subprocess.check_output(["java", "-cp", temp_dir, class_name], stderr=subprocess.STDOUT, timeout=5)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

def calculate_codebleu(reference, prediction, lang="java"):
    try:
        result = calc_codebleu([reference], [prediction], lang=lang, weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
        return result
    except Exception as e:
        return {"codebleu": 0.0, "ngram_match_score": 0.0, "weighted_ngram_match_score": 0.0, "syntax_match_score": 0.0, "dataflow_match_score": 0.0}

def generate_and_evaluate(data, tokenizer, model, device):
    results = []
    
    print(f"Evaluating {len(data)} examples...")
    
    for item in tqdm(data):
        python_code = item['python']
        reference_java = item['java']
        
        input_text = "Translate Python to Java: " + python_code
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=256, 
            truncation=True, 
            padding="max_length"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs, 
                max_length=256, # Increased length
                num_beams=5, 
                early_stopping=True
            )
        
        prediction = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        # 1. Compilation
        is_compiled = check_compilation(prediction)
        
        # 2. Execution
        is_executed = check_execution(prediction)
        
        # 3. CodeBLEU
        cb_metrics = calculate_codebleu(reference_java, prediction)
        
        results.append({
            "python": python_code,
            "reference": reference_java,
            "prediction": prediction,
            "compiled": is_compiled,
            "executed": is_executed,
            "codebleu": cb_metrics['codebleu'],
            "syntax_match": cb_metrics['syntax_match_score'],
            "dataflow_match": cb_metrics['dataflow_match_score']
        })
        
    return results

def main():
    tokenizer, model, device = load_model()
    data = load_data(DATA_PATH)
    
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
    execution_rate = df['executed'].mean()
    
    print("\n--- Easy Evaluation Results ---")
    print(f"BLEU Score: {bleu_score:.2f}")
    print(f"CodeBLEU: {avg_codebleu:.4f}")
    print(f"  - Syntax Match: {avg_syntax:.4f}")
    print(f"  - Dataflow Match: {avg_dataflow:.4f}")
    print(f"Compilation Rate: {compilation_rate:.2%}")
    print(f"Execution Rate: {execution_rate:.2%}")
    
    # Save results
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDetailed results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
