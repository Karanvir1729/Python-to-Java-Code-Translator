# CodeT5 Python to Java Translator

This package contains a fine-tuned CodeT5-small model for translating Python code to Java.

## Contents
- `codet5_fine_tuned/`: The fine-tuned model weights and tokenizer.
- `test_codet5.py`: An interactive script to test the model.

## How to Run
1.  Ensure you have Python installed.
2.  Install the required dependencies:
    ```bash
    pip install torch transformers
    ```
    *(Note: `torch` installation depends on your OS/Hardware. See https://pytorch.org/)*

3.  Run the inference script:
    ```bash
    python test_codet5.py
    ```

4.  Enter Python code when prompted, and the model will output the Java translation.

## Model Details
- **Base Model**: `Salesforce/codet5-small`
- **Dataset**: BigCodeNet (~15k Python-Java pairs)
- **Fine-tuning**: Trained for 3 epochs.
