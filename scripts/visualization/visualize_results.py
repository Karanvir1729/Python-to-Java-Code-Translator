import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "../../results")
TABLES_DIR = os.path.join(BASE_DIR, "../../tables")
EASY_FILE = "easy_evaluation_results.csv"
ADVANCED_FILE = "advanced_evaluation_results.csv"
BIGCODENET_FILE = "bigcodenet_evaluation_results.csv"

def load_data():
    datasets = {}
    files = {
        "Easy": EASY_FILE,
        "Advanced": ADVANCED_FILE,
        "BigCodeNet": BIGCODENET_FILE
    }
    
    for name, filename in files.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Ensure columns exist
                if 'codebleu' not in df.columns: df['codebleu'] = 0.0
                if 'compiled' not in df.columns: df['compiled'] = False
                if 'prediction' not in df.columns: df['prediction'] = ""
                datasets[name] = df
            except Exception as e:
                print(f"Error loading {name}: {e}")
    return datasets

def generate_summary_table(datasets):
    summary_data = []
    
    for name, df in datasets.items():
        metrics = {
            "Dataset": name,
            "Samples": len(df),
            "Compilation Rate (%)": df['compiled'].mean() * 100,
            "Avg CodeBLEU": df['codebleu'].mean(),
            "Avg Syntax Match": df['syntax_match'].mean() if 'syntax_match' in df.columns else 0.0,
            "Avg Dataflow Match": df['dataflow_match'].mean() if 'dataflow_match' in df.columns else 0.0
        }
        summary_data.append(metrics)
        
        # Save detailed table
        df.to_csv(os.path.join(TABLES_DIR, f"detailed_{name.lower()}.csv"), index=False)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(TABLES_DIR, "summary_metrics.csv"), index=False)
    print("Summary table saved.")
    return summary_df

def plot_compilation_rates(summary_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Dataset", y="Compilation Rate (%)", data=summary_df, palette="viridis")
    plt.title("Compilation Rate by Dataset")
    plt.ylim(0, 100)
    plt.ylabel("Compilation Rate (%)")
    plt.savefig(os.path.join(TABLES_DIR, "compilation_rate_comparison.png"))
    plt.close()

def plot_codebleu_scores(summary_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Dataset", y="Avg CodeBLEU", data=summary_df, palette="magma")
    plt.title("Average CodeBLEU Score by Dataset")
    plt.ylim(0, 1.0) # Assuming CodeBLEU is 0-1
    plt.ylabel("CodeBLEU Score")
    plt.savefig(os.path.join(TABLES_DIR, "codebleu_score_comparison.png"))
    plt.close()

def plot_length_distributions(datasets):
    # 1. Combined Prediction Lengths (Existing)
    plt.figure(figsize=(12, 6))
    for name, df in datasets.items():
        lengths = df['prediction'].astype(str).apply(len)
        sns.kdeplot(lengths, label=name, fill=True, alpha=0.3)
    
    plt.title("Prediction Code Length Distribution Comparison")
    plt.xlabel("Length (characters)")
    plt.legend()
    plt.savefig(os.path.join(TABLES_DIR, "code_length_distribution_all_predictions.png"))
    plt.close()

    # 2. Detailed Source vs Ref vs Pred for EACH dataset
    for name, df in datasets.items():
        plt.figure(figsize=(12, 6))
        
        # Calculate lengths
        len_source = df['python'].astype(str).apply(len)
        len_ref = df['reference'].astype(str).apply(len)
        len_pred = df['prediction'].astype(str).apply(len)
        
        sns.kdeplot(len_source, label='Source (Python)', fill=True, alpha=0.3, color='blue')
        sns.kdeplot(len_ref, label='Reference (Java)', fill=True, alpha=0.3, color='green')
        sns.kdeplot(len_pred, label='Prediction (Java)', fill=True, alpha=0.3, color='red')
        
        plt.title(f"Code Length Distribution - {name}")
        plt.xlabel("Length (characters)")
        plt.legend()
        plt.savefig(os.path.join(TABLES_DIR, f"code_length_distribution_detailed_{name.lower()}.png"))
        plt.close()

def main():
    if not os.path.exists(TABLES_DIR):
        os.makedirs(TABLES_DIR)
        
    datasets = load_data()
    if not datasets:
        print("No data found.")
        return

    summary_df = generate_summary_table(datasets)
    
    # Generate Plots
    plot_compilation_rates(summary_df)
    plot_codebleu_scores(summary_df)
    plot_length_distributions(datasets)
    
    print(f"Visualizations generated in {TABLES_DIR}/")

if __name__ == "__main__":
    main()
