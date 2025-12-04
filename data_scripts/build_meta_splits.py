import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Max number of solutions per language to retain per problem
# Adjust if needed; this caps the balanced per-problem count.
MAX_SOLUTIONS_PER_LANG_PER_PROBLEM = 5

# Target number of solutions per language to keep per problem
# Problems with fewer than this per-language count are excluded from splits.
TARGET_SOLUTIONS_PER_LANG_PER_PROBLEM = 3


def load_accepted_lang_paths(base_path: Path) -> Dict[str, Dict[str, List[Tuple[str, Path]]]]:
    """
    Read metadata CSVs and return, for each problem, accepted Python and Java solutions
    with their filenames and absolute paths to the source files under `data/`.

    Returns a nested dict:
      { problem_id: { 'Python': [(filename, path), ...], 'Java': [(filename, path), ...] } }
    Only includes entries where the corresponding file exists.
    """
    metadata_dir = base_path / 'metadata'
    data_dir = base_path / 'data'

    per_problem: Dict[str, Dict[str, List[Tuple[str, Path]]]] = {}

    # Iterate over problem metadata CSVs
    for meta_csv in sorted(metadata_dir.glob('p?????.csv')):
        problem_id = meta_csv.stem

        # Ensure data folders exist for both languages for this problem
        py_dir = data_dir / problem_id / 'Python'
        java_dir = data_dir / problem_id / 'Java'
        if not (py_dir.is_dir() and java_dir.is_dir()):
            continue

        # Prepare collections
        lang_map: Dict[str, List[Tuple[str, Path]]] = {'Python': [], 'Java': []}

        # Parse metadata rows
        with meta_csv.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lang = row.get('language')
                status = row.get('status')
                ext = row.get('filename_ext')
                sub_id = row.get('submission_id')

                # Keep only Python/Java that are Accepted (exclude inaccurate)
                if lang not in ('Python', 'Java'):
                    continue
                if status != 'Accepted':
                    continue

                if not sub_id or not ext:
                    continue

                filename = f"{sub_id}.{ext}"
                if lang == 'Python':
                    src_path = py_dir / filename
                else:  # Java
                    src_path = java_dir / filename

                if src_path.is_file():
                    lang_map[lang].append((filename, src_path))

        # Sort for deterministic selection
        for k in lang_map:
            lang_map[k].sort(key=lambda t: t[0])

        # Keep only problems with at least one accepted in both languages
        if lang_map['Python'] and lang_map['Java']:
            per_problem[problem_id] = lang_map

    return per_problem


def split_problems(problem_ids: List[str], ratios=(0.7, 0.15, 0.15)) -> Tuple[List[str], List[str], List[str]]:
    """
    Deterministically split problems by sorted order into train/test/valid by ratios.
    """
    problem_ids = sorted(problem_ids)
    n = len(problem_ids)
    n_train = int(n * ratios[0])
    n_test = int(n * ratios[1])
    # Remainder goes to valid to ensure coverage
    n_valid = n - n_train - n_test

    train = problem_ids[:n_train]
    test = problem_ids[n_train:n_train + n_test]
    valid = problem_ids[n_train + n_test:]
    return train, test, valid


def write_split_csv(
    base_path: Path,
    per_problem: Dict[str, Dict[str, List[Tuple[str, Path]]]],
    problem_split: List[str],
    out_csv: Path,
):
    """
    For a given list of problems, choose the same number K of solutions per language
    for every problem in the split, where K is the minimum across problems of
    min(len(py), len(java)).
    Writes rows with columns:
      problem_id, language, filename, relative_path, count_for_problem
    """
    if not problem_split:
        # Create an empty CSV with header for completeness
        with out_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["problem_id", "language", "filename", "relative_path", "count_for_problem"])
        return

    # Fixed K per problem derived from target and max cap
    if not problem_split:
        K = 0
    else:
        K = min(TARGET_SOLUTIONS_PER_LANG_PER_PROBLEM, MAX_SOLUTIONS_PER_LANG_PER_PROBLEM)

    # If K is zero, we cannot pick balanced solutions; produce header-only file
    if K <= 0:
        with out_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["problem_id", "language", "filename", "relative_path", "count_for_problem"])
        return

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["problem_id", "language", "filename", "relative_path", "count_for_problem"])

        for pid in problem_split:
            lang_map = per_problem[pid]
            # Safety: skip any problem that does not meet K after filtering
            if len(lang_map['Python']) < K or len(lang_map['Java']) < K:
                continue
            py_list = lang_map['Python'][:K]
            java_list = lang_map['Java'][:K]

            for filename, abs_path in py_list:
                rel = str(abs_path.relative_to(base_path))
                writer.writerow([pid, 'Python', filename, rel, K])

            for filename, abs_path in java_list:
                rel = str(abs_path.relative_to(base_path))
                writer.writerow([pid, 'Java', filename, rel, K])


def build_meta_splits(base_path: Path, output_dir: Path):
    """
    Build training_meta.csv (70%), test_meta.csv (15%), valid_meta.csv (15%) using
    problem-level splitting, excluding inaccurate solutions, and ensuring each problem
    contains the same number of Python and Java solutions and the same total count
    per problem within each split.
    """
    per_problem_all = load_accepted_lang_paths(base_path)

    # Enforce minimum availability per language (target) and keep only eligible problems
    problem_ids = sorted(
        pid
        for pid, langs in per_problem_all.items()
        if len(langs.get('Python', [])) >= TARGET_SOLUTIONS_PER_LANG_PER_PROBLEM
        and len(langs.get('Java', [])) >= TARGET_SOLUTIONS_PER_LANG_PER_PROBLEM
    )
    if not problem_ids:
        raise RuntimeError("No eligible problems found with accepted Python and Java solutions.")

    train_ids, test_ids, valid_ids = split_problems(problem_ids)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a filtered per_problem map limited to eligible problems only
    per_problem = {pid: per_problem_all[pid] for pid in problem_ids}

    write_split_csv(base_path, per_problem, train_ids, output_dir / 'training_meta.csv')
    write_split_csv(base_path, per_problem, test_ids, output_dir / 'test_meta.csv')
    write_split_csv(base_path, per_problem, valid_ids, output_dir / 'valid_meta.csv')

    print(f"[OK] Wrote training_meta.csv with {len(train_ids)} problems -> {output_dir / 'training_meta.csv'}")
    print(f"[OK] Wrote test_meta.csv with {len(test_ids)} problems -> {output_dir / 'test_meta.csv'}")
    print(f"[OK] Wrote valid_meta.csv with {len(valid_ids)} problems -> {output_dir / 'valid_meta.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build meta CSV splits (train/test/valid) from metadata, splitting by problems, "
            "excluding inaccurate solutions, and balancing Python/Java counts per problem."
        )
    )
    parser.add_argument(
        "--base_path",
        default=str(Path(__file__).resolve().parents[1]),
        help="Base path containing 'metadata/' and 'data/' (default: project root)",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Directory to write training_meta.csv, test_meta.csv, valid_meta.csv (default: project root)",
    )
    args = parser.parse_args()

    base_path = Path(args.base_path)
    output_dir = Path(args.output_dir)
    build_meta_splits(base_path, output_dir)


if __name__ == '__main__':
    main()
