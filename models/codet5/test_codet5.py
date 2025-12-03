import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import os

MODEL_PATH = "./codet5_fine_tuned"
BASE_MODEL = "Salesforce/codet5-small"

def load_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading fine-tuned model from {MODEL_PATH}...")
        model_name = MODEL_PATH
    else:
        print(f"Fine-tuned model not found at {MODEL_PATH}")
        print(f"Loading base model {BASE_MODEL} for demonstration...")
        print("NOTE: Base model might not translate perfectly without fine-tuning.")
        model_name = BASE_MODEL

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Device setup
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    model.to(device)
    return tokenizer, model, device

def translate(python_code, tokenizer, model, device):
    # Add the same prefix used in training
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
    
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

if __name__ == "__main__":
    tokenizer, model, device = load_model()
    print(f"Running on {device}")
    print("\n--- CodeT5 Python to Java Translator ---")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            print("Enter Python code (press Enter twice to submit):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            
            python_code = "\n".join(lines).strip()
            
            if not python_code:
                continue
                
            if python_code.lower() in ['exit', 'quit']:
                break
            
            java_code = translate(python_code, tokenizer, model, device)
            
            print("\n>>> Java Translation:")
            print(java_code)
            print("-" * 30 + "\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
