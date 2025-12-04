import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import json
import pandas as pd
import evaluate
from tqdm import tqdm
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../models/codet5_fine_tuned")
DATA_PATH = os.path.join(BASE_DIR, "../../data/advanced_code_pairs.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "../../results/evaluation_results.csv")
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

def generate_translations(data, tokenizer, model, device):
    results = []
    
    print(f"Generating translations for {len(data)} examples...")
    
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
        
        results.append({
            "python": python_code,
            "reference": reference_java,
            "prediction": prediction
        })
        
    return results

def calculate_metrics(results):
    bleu = evaluate.load("sacrebleu")
    
    predictions = [r['prediction'] for r in results]
    references = [[r['reference']] for r in results] # sacrebleu expects list of lists for references
    
    # BLEU
    bleu_score = bleu.compute(predictions=predictions, references=references)
    
    # Exact Match
    exact_matches = sum(1 for r in results if r['prediction'].strip() == r['reference'].strip())
    exact_match_score = exact_matches / len(results)
    
    return bleu_score, exact_match_score

def main():
    tokenizer, model, device = load_model()
    data = load_data(DATA_PATH)
    
    # Use a subset for quicker testing if needed, or full dataset
    # data = data[:20] 
    
    results = generate_translations(data, tokenizer, model, device)
    
    bleu_score, exact_match_score = calculate_metrics(results)
    
    print("\n--- Evaluation Results ---")
    print(f"BLEU Score: {bleu_score['score']:.2f}")
    print(f"Exact Match: {exact_match_score:.2%}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDetailed results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
