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
DATA_PATH = os.path.join(BASE_DIR, "../../data/advanced_code_pairs.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "../../results/advanced_evaluation_results.csv")
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
    # Simple heuristic: if it doesn't look like a class, wrap it
    code_to_compile = java_code
    if "public class" not in java_code and "class " not in java_code:
        code_to_compile = f"public class TempClass {{\n{java_code}\n}}"
        class_name = "TempClass"
    else:
        # Try to extract class name or default to Main if we can't find it easily
        # This is a simplification; for robust parsing we'd need regex or AST
        if "public class Main" in java_code:
            class_name = "Main"
        else:
            # If it's a full class but not Main, we might have issues with filename matching
            # For this script, we'll try to save as Main.java and hope the class is Main or non-public
            class_name = "Main" 
            # If the code has a public class named something else, javac will complain about filename
            # So we might need to parse the class name. 
            # Let's stick to the wrapper approach for snippets which is most of the dataset.
            pass

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
    """
    Wraps code in a class with main method and checks if it runs.
    Returns True if runs without error, False otherwise.
    """
    # We need a main method to run. 
    # If the code is just a function, we need to call it from main.
    # This is hard to automate generically without knowing function signature.
    # For now, we will only check execution if the code ALREADY contains a main method.
    # OR if we can wrap it effectively. 
    
    # Strategy: If it has "public static void main", try to run it.
    if "public static void main" in java_code:
        class_name = "Main" # Assumption for simplicity
        if "class " not in java_code:
             code_to_run = f"public class {class_name} {{\n{java_code}\n}}"
        else:
             # If it has class, we need to match filename. 
             # Let's try to find class name or force it to be Main if possible.
             # If user provided class is not Main, we can't easily run it without parsing.
             # We'll try to save as Main.java.
             code_to_run = java_code
             if "public class" in java_code:
                 # Extract name
                 try:
                     parts = java_code.split("public class")[1].strip().split()
                     class_name = parts[0].split("{")[0].strip()
                 except:
                     pass

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
    
    return False # Not runnable or failed

def calculate_codebleu(reference, prediction, lang="java"):
    try:
        result = calc_codebleu([reference], [prediction], lang=lang, tokenizer=None, params="0.25,0.25,0.25,0.25")
        return result
    except Exception as e:
        # Fallback if codebleu fails (e.g. syntax error in parsing)
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
            max_length=128, 
            truncation=True, 
            padding="max_length"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs, 
                max_length=128, 
                num_beams=5, 
                early_stopping=True
            )
        
        prediction = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        # 1. Compilation
        is_compiled = check_compilation(prediction)
        
        # 2. Execution (Best Effort)
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
    
    # Use a subset for quicker testing if needed, or full dataset
    data = data[:20] 
    
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
    
    print("\n--- Advanced Evaluation Results ---")
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
