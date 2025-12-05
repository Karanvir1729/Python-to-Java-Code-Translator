import pandas as pd
import re
import os
import sys
import io

# Set stdout to handle utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def extract_logic_signature(code, lang="python"):
    """Extracts a 'Logic Signature' representing the functional behavior of the code."""
    if not isinstance(code, str): return {
        "has_input": False, "has_output": False, "loops": 0, "conditionals": 0, "math": set()
    }
    
    sig = {
        "has_input": False,
        "has_output": False,
        "has_return": False,
        "loops": 0,
        "conditionals": 0,
        "math": set()
    }
    
    # Math Operators
    for op in ['+', '-', '*', '/', '%']:
        if op in code: sig["math"].add(op)
        
    if lang == "python":
        # Input (Explicit or Implicit via Args)
        if re.search(r'\binput\(|\bsys\.stdin', code): sig["has_input"] = True
        if re.search(r'def\s+\w+\s*\([^)]+\)', code): sig["has_args"] = True # Function with args
        
        # Output
        if re.search(r'\bprint\(', code): sig["has_output"] = True
        # Return
        if re.search(r'\breturn\b', code): sig["has_return"] = True
        # Loops
        sig["loops"] = len(re.findall(r'\bfor\b|\bwhile\b', code))
        # Conditionals
        sig["conditionals"] = len(re.findall(r'\bif\b', code))
        
    else: # Java
        # Input
        if re.search(r'Scanner|BufferedReader|System\.in', code): sig["has_input"] = True
        # Output
        if re.search(r'System\.out\.print', code): sig["has_output"] = True
        # Return
        if re.search(r'\breturn\b', code): sig["has_return"] = True
        # Loops
        sig["loops"] = len(re.findall(r'\bfor\b|\bwhile\b', code))
        # Conditionals
        sig["conditionals"] = len(re.findall(r'\bif\b|\bswitch\b', code))
        
    return sig

def calculate_deep_score(row):
    py_code = row['Python Code']
    pt_code = row['Pre-trained Prediction']
    ft_code = row['Fine-tuned Prediction']
    
    py_sig = extract_logic_signature(py_code, "python")
    
    def get_recall_score(pred_code, lang="java"):
        pred_sig = extract_logic_signature(pred_code, lang)
        
        score = 0.0
        total_weight = 0.0
        
        # 1. Data Flow (Input) - Weight 1
        # Python Input OR Args -> Java Input
        has_py_in = py_sig["has_input"] or py_sig.get("has_args", False)
        if has_py_in:
            total_weight += 1.0
            if pred_sig["has_input"]: score += 1.0
            
        # 2. Data Flow (Output) - Weight 1
        # Python Print OR Return -> Java Print OR Return
        has_py_out = py_sig["has_output"] or py_sig["has_return"]
        has_java_out = pred_sig["has_output"] or pred_sig["has_return"]
        
        if has_py_out:
            total_weight += 1.0
            if has_java_out: score += 1.0

        # 3. Control Flow (Weight 2) - Loops
        if py_sig["loops"] > 0:
            total_weight += 2.0
            if pred_sig["loops"] == py_sig["loops"]: score += 2.0
            elif pred_sig["loops"] > 0: score += 1.0
            
        # 4. Control Flow (Weight 2) - Conditionals
        if py_sig["conditionals"] > 0:
            total_weight += 2.0
            if pred_sig["conditionals"] == py_sig["conditionals"]: score += 2.0
            elif pred_sig["conditionals"] > 0: score += 1.0
            
        # 5. Math Logic (Weight 1)
        if len(py_sig["math"]) > 0:
            total_weight += 1.0
            common_ops = py_sig["math"].intersection(pred_sig["math"])
            if len(common_ops) == len(py_sig["math"]): score += 1.0
            elif len(common_ops) > 0: score += 0.5
            
        if total_weight == 0:
            noise = pred_sig["loops"] + pred_sig["conditionals"]
            return 1.0 if noise == 0 else 0.5
            
        return score / total_weight

    pt_score = get_recall_score(pt_code)
    ft_score = get_recall_score(ft_code)
    
    return pd.Series({
        "Pre-trained Logic Score": round(pt_score, 2),
        "Fine-tuned Logic Score": round(ft_score, 2)
    })

def main():
    file_path = "results/comparison_10_samples.csv"
    if not os.path.exists(file_path):
        print("File not found.")
        return
        
    df = pd.read_csv(file_path)
    
    # Apply scoring
    score_cols = df.apply(calculate_deep_score, axis=1)
    df = pd.concat([df, score_cols], axis=1)
    
    # Print Averages
    print("\n### Deep Logic Recall Score (Pre-trained vs Fine-tuned)")
    avg_df = df.groupby("Dataset")[["Pre-trained Logic Score", "Fine-tuned Logic Score"]].mean()
    print(avg_df.to_markdown())
    
    # Save detailed CSV
    df.to_csv("results/deep_logic_evaluation.csv", index=False)
    print("\nSaved detailed results to results/deep_logic_evaluation.csv")

if __name__ == "__main__":
    main()
