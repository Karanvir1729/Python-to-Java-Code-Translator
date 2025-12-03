# 413 Group Assignment - CodeT5 Model

This repository contains the code, data, and results for the CodeT5 model used in our group assignment.

## Directory Structure

- **`data/`**: Contains the dataset files used for training and evaluation.
  - `code_pairs.json`: Main dataset of code pairs.
  - `easy_code_pairs.json`: Subset for easy evaluation.

- **`models/`**: Contains the model artifacts and source code.
  - `codet5/`: **(Main Model)** The primary CodeT5 model used for the assignment.
  - `ansh_model/`: Experimental model variants.
  - `codet5_fine_tuned/`: Experimental fine-tuned versions of CodeT5.

- **`scripts/`**: Contains scripts for training, evaluation, and visualization.
  - `training/`: Scripts for training the model (e.g., `train_codet5_cpu.py`).
  - `evaluation/`: Scripts for evaluating the model on different datasets (e.g., `evaluate_codet5.py`, `evaluate_easy.py`).
  - `visualization/`: Scripts for generating plots and visualizing results (e.g., `visualize_results.py`).

- **`results/`**: Contains the output results from evaluations.
  - `evaluation_results.csv`: General evaluation metrics.
  - `*_results.csv`: Specific results for different test sets.

## Usage

### Setup & Model Weights
To try out the model, please download the weights from the following link:
[Download Model Weights](https://drive.google.com/file/d/1CLwoeOP3yS4yfH4RRovssxQRd--ri__x/view?usp=drive_link)

1. Unzip the downloaded file.
2. Ensure the unzipped files are placed in `models/codet5/`.
3. Run `test_codet5.py` to verify:
   ```bash
   python scripts/evaluation/test_codet5.py
   ```
   *(Note: You might need to adjust the path in the script if it differs)*

### Training
To train the model, use the script in `scripts/training/`:
```bash
python scripts/training/train_codet5_cpu.py
```

### Evaluation
To evaluate the model, run the scripts in `scripts/evaluation/`:
```bash
python scripts/evaluation/evaluate_codet5.py
```

### Visualization
To generate visualization plots:
```bash
python scripts/visualization/visualize_results.py
```
