
import pandas as pd
import json
from tqdm import tqdm
import argparse

def create_translation_pairs(csv_file, num_problems=10):
    """
    Parses the metadata CSV and creates pairs of Python and Java files for translation.

    Args:
        csv_file (str): The path to the metadata CSV file.
        num_problems (int): The number of problems to process.

    Returns:
        list: A list of tuples, where each tuple contains the path to the Python file
              and the path to the Java file for a given problem.
    """
    df = pd.read_csv(csv_file)
    pairs = []
    for problem_id in tqdm(df["problem_id"].unique()[:num_problems]):
        problem_df = df[df["problem_id"] == problem_id]
        python_files = problem_df[problem_df["language"] == "Python"]["relative_path"].tolist()
        java_files = problem_df[problem_df["language"] == "Java"]["relative_path"].tolist()

        # For simplicity, we'll just take the first Python and Java file for each problem.
        if python_files and java_files:
            pairs.append((python_files[0], java_files[0]))
    return pairs

def prepare_data(pairs, root_dir="/Users/karanvirkhanna/BigCodeNet"):
    """
    Reads the content of the file pairs and prepares the data for training.

    Args:
        pairs (list): A list of tuples, where each tuple contains the relative path to the
                      Python file and the relative path to the Java file.
        root_dir (str): The root directory of the project.

    Returns:
        list: A list of dictionaries, where each dictionary has two keys: "input_text"
              and "target_text".
    """
    data = []
    for python_file, java_file in tqdm(pairs):
        try:
            with open(f"{root_dir}/{python_file}", "r") as f:
                python_code = f.read()
            with open(f"{root_dir}/{java_file}", "r") as f:
                java_code = f.read()

            data.append({
                "input_text": f"translate Python to Java: {python_code}",
                "target_text": java_code
            })
        except FileNotFoundError:
            print(f"Could not find file: {python_file} or {java_file}")
            continue
    return data

if __name__ == "__main__":
    validation_pairs = create_translation_pairs("/Users/karanvirkhanna/BigCodeNet/valid_meta.csv")
    validation_data = prepare_data(validation_pairs)

    with open("/Users/karanvirkhanna/BigCodeNet/model/validation_data.json", "w") as f:
        json.dump(validation_data, f, indent=4)

    print(f"Prepared {len(validation_data)} validation examples.")
