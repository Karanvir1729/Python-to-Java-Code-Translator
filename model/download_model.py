
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def download_model():
    """Downloads the T5-base model and tokenizer from Hugging Face and saves them locally."""
    model_name = "t5-base"
    
    # Create the save directory if it doesn't exist
    save_directory = "./model_files"
    os.makedirs(save_directory, exist_ok=True)

    # Download and save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)

    # Download and save the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.save_pretrained(save_directory)
    
    print(f"Model and tokenizer saved to: {os.path.abspath(save_directory)}")

if __name__ == "__main__":
    download_model()
