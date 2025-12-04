import csv
import argparse
from pathlib import Path

# Build a meta CSV file listing problem solutions in Python and Java, ensuring equal number of solutions per language for each problem.
# Run the script with: python build_meta_csv.py --data_root <path_to_codenet_data_folder> --output_csv meta.csv

def build_meta_csv(data_root, output_csv):
    data_root = Path(data_root)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["problem_id", "language", "filename", "relative_path", "count_for_problem"]
        )

        # loop over problems
        for problem_path in sorted(data_root.iterdir()):
            if not problem_path.is_dir():
                continue

            pid = problem_path.name

            py_dir = problem_path / "Python"
            java_dir = problem_path / "Java"

            # must have both languages
            if not py_dir.is_dir() or not java_dir.is_dir():
                continue

            # take ALL files in each folder 
            py_files = sorted(p for p in py_dir.rglob("*") if p.is_file())
            java_files = sorted(j for j in java_dir.rglob("*") if j.is_file())

            if not py_files or not java_files:
                continue

            # same number of python + java solutions
            k = min(len(py_files), len(java_files))

            py_keep = py_files[:k]
            java_keep = java_files[:k]

            # Python rows
            for p in py_keep:
                rel = str(p.relative_to(data_root)) 
                writer.writerow([pid, "Python", p.name, rel, k])

            # Java rows
            for j in java_keep:
                rel = str(j.relative_to(data_root))  
                writer.writerow([pid, "Java", j.name, rel, k])

    print(f"[OK] Saved meta CSV to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Build meta CSV with equal-number Python & Java solutions per problem."
    )
    parser.add_argument("--data_root", required=True,
                        help="Path to CodeNet data folder (contains p00000/, p00001/, ...)")
    parser.add_argument("--output_csv", default="meta.csv",
                        help="Output CSV path (default: meta.csv)")
    args = parser.parse_args()

    build_meta_csv(args.data_root, args.output_csv)


if __name__ == "__main__":
    main()
