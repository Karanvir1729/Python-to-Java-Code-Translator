
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def translate_code(python_code):
    """
    Translates a Python code snippet to Java using the fine-tuned model.

    Args:
        python_code (str): The Python code to translate.

    Returns:
        str: The translated Java code.
    """
    # Load the fine-tuned model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained("/Users/karanvirkhanna/BigCodeNet/model/fine-tuned-model")
    model = T5ForConditionalGeneration.from_pretrained("/Users/karanvirkhanna/BigCodeNet/model/fine-tuned-model")

    # Prepare the input text
    input_text = f"translate Python to Java: {python_code}"

    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Generate the translation
    outputs = model.generate(input_ids, max_length=512)
    translated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_code

if __name__ == "__main__":
    python_code = """
def greet(name):
    print(f"Hello, {name}!")
"""
    java_code = translate_code(python_code)
    print("Python code:")
    print(python_code)
    print("\nTranslated Java code:")
    print(java_code)

