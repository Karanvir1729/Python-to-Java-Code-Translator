#!/usr/bin/env python3
"""
Script to analyze Project CodeNet metadata and compute statistics
for Python and Java submissions, including Cartesian product calculations.
"""

import csv
import os
from pathlib import Path
from collections import defaultdict
import json

def analyze_metadata(base_path='/Users/karanvirkhanna/BigCodeNet'):
    """Analyze all metadata CSV files to get submission statistics."""
    
    metadata_dir = Path(base_path) / 'metadata'
    
    # Load problem list
    problem_count = 0
    with open(metadata_dir / 'problem_list.csv', 'r') as f:
        reader = csv.DictReader(f)
        problem_count = sum(1 for _ in reader)
    print(f"Total problems in problem_list.csv: {problem_count}")
    
    # Statistics containers
    total_submissions = 0
    python_submissions = 0
    java_submissions = 0
    
    language_counts = defaultdict(int)
    status_counts = defaultdict(int)
    
    python_accepted = 0
    java_accepted = 0
    
    problems_with_python = 0
    problems_with_java = 0
    problems_with_both = 0
    
    python_by_problem = {}
    java_by_problem = {}
    
    problem_stats = []
    
    # Get all problem CSV files
    problem_csvs = sorted(metadata_dir.glob('p?????.csv'))
    
    print(f"\nAnalyzing {len(problem_csvs)} problem metadata files...")
    
    for i, csv_file in enumerate(problem_csvs):
        problem_id = csv_file.stem
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(problem_csvs)} problems...")
        
        try:
            submissions = []
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                submissions = list(reader)
            
            # Count submissions
            problem_total = len(submissions)
            total_submissions += problem_total
            
            # Count by language
            python_in_problem = sum(1 for s in submissions if s['language'] == 'Python')
            java_in_problem = sum(1 for s in submissions if s['language'] == 'Java')
            
            python_submissions += python_in_problem
            java_submissions += java_in_problem
            
            # Count accepted submissions
            python_accepted_in_problem = sum(1 for s in submissions 
                                            if s['language'] == 'Python' and s['status'] == 'Accepted')
            java_accepted_in_problem = sum(1 for s in submissions 
                                          if s['language'] == 'Java' and s['status'] == 'Accepted')
            
            python_accepted += python_accepted_in_problem
            java_accepted += java_accepted_in_problem
            
            # Track problems with each language
            if python_in_problem > 0:
                problems_with_python += 1
                python_by_problem[problem_id] = python_in_problem
            
            if java_in_problem > 0:
                problems_with_java += 1
                java_by_problem[problem_id] = java_in_problem
            
            if python_in_problem > 0 and java_in_problem > 0:
                problems_with_both += 1
            
            # Overall language counts
            for s in submissions:
                language_counts[s['language']] += 1
                status_counts[s['status']] += 1
            
            # Store problem-level stats
            problem_stats.append({
                'problem_id': problem_id,
                'total': problem_total,
                'python': python_in_problem,
                'java': java_in_problem,
                'python_accepted': python_accepted_in_problem,
                'java_accepted': java_accepted_in_problem
            })
            
        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")
            continue
    
    print(f"  Completed processing all {len(problem_csvs)} problems!\n")
    
    # Calculate Cartesian product
    cartesian_product_size = python_submissions * java_submissions
    
    # Calculate problem-level Cartesian products
    problem_level_cartesian = 0
    for problem_id in python_by_problem:
        if problem_id in java_by_problem:
            problem_level_cartesian += python_by_problem[problem_id] * java_by_problem[problem_id]
    
    # Sort languages by count
    sorted_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_statuses = sorted(status_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create results dictionary
    results = {
        'total_problems': len(problem_csvs),
        'total_submissions': total_submissions,
        'python_submissions': python_submissions,
        'java_submissions': java_submissions,
        'python_accepted': python_accepted,
        'java_accepted': java_accepted,
        'python_acceptance_rate': python_accepted / python_submissions if python_submissions > 0 else 0,
        'java_acceptance_rate': java_accepted / java_submissions if java_submissions > 0 else 0,
        'problems_with_python': problems_with_python,
        'problems_with_java': problems_with_java,
        'problems_with_both': problems_with_both,
        'cartesian_product_all': cartesian_product_size,
        'cartesian_product_within_problems': problem_level_cartesian,
        'language_distribution': sorted_languages,
        'status_distribution': sorted_statuses,
        'problem_stats': problem_stats
    }
    
    return results

def format_large_number(n):
    """Format large numbers with commas for readability."""
    return f"{n:,}"

def write_summary_report(results, output_file='metadataSummary.txt'):
    """Write a comprehensive summary report."""
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PROJECT CODENET METADATA ANALYSIS SUMMARY\n")
        f.write("Analysis of Python and Java Submissions\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall Statistics
        f.write("OVERALL DATASET STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Problems:           {format_large_number(results['total_problems'])}\n")
        f.write(f"Total Submissions:        {format_large_number(results['total_submissions'])}\n")
        f.write(f"\n")
        
        # Python Statistics
        f.write("PYTHON SUBMISSIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Python Submissions: {format_large_number(results['python_submissions'])}\n")
        f.write(f"Accepted Python:          {format_large_number(results['python_accepted'])}\n")
        f.write(f"Python Acceptance Rate:   {results['python_acceptance_rate']:.2%}\n")
        f.write(f"Problems with Python:     {format_large_number(results['problems_with_python'])}\n")
        python_pct = (results['python_submissions'] / results['total_submissions']) * 100
        f.write(f"Python % of Total:        {python_pct:.2f}%\n")
        f.write(f"\n")
        
        # Java Statistics
        f.write("JAVA SUBMISSIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Java Submissions:   {format_large_number(results['java_submissions'])}\n")
        f.write(f"Accepted Java:            {format_large_number(results['java_accepted'])}\n")
        f.write(f"Java Acceptance Rate:     {results['java_acceptance_rate']:.2%}\n")
        f.write(f"Problems with Java:       {format_large_number(results['problems_with_java'])}\n")
        java_pct = (results['java_submissions'] / results['total_submissions']) * 100
        f.write(f"Java % of Total:          {java_pct:.2f}%\n")
        f.write(f"\n")
        
        # Cross-language Statistics
        f.write("CROSS-LANGUAGE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Problems with Both:       {format_large_number(results['problems_with_both'])}\n")
        f.write(f"Problems Python Only:     {format_large_number(results['problems_with_python'] - results['problems_with_both'])}\n")
        f.write(f"Problems Java Only:       {format_large_number(results['problems_with_java'] - results['problems_with_both'])}\n")
        f.write(f"\n")
        
        # Cartesian Product Calculations
        f.write("CARTESIAN PRODUCT CALCULATIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Global Cartesian Product:\n")
        f.write(f"  (All Python × All Java)\n")
        f.write(f"  {format_large_number(results['python_submissions'])} × {format_large_number(results['java_submissions'])} = \n")
        f.write(f"  {format_large_number(results['cartesian_product_all'])} pairs\n")
        f.write(f"\n")
        
        f.write(f"Problem-Level Cartesian Product:\n")
        f.write(f"  (Sum of Python × Java within each problem)\n")
        f.write(f"  Total pairs: {format_large_number(results['cartesian_product_within_problems'])}\n")
        f.write(f"\n")
        
        f.write(f"Note: The problem-level Cartesian product is more meaningful for\n")
        f.write(f"      pair analysis as it only pairs submissions to the same problem.\n")
        f.write(f"\n")
        
        # Language Distribution
        f.write("LANGUAGE DISTRIBUTION (Top 15)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Language':<20} {'Submissions':<15} {'Percentage':<10}\n")
        f.write("-" * 80 + "\n")
        
        for i, (lang, count) in enumerate(results['language_distribution'][:15], 1):
            pct = (count / results['total_submissions']) * 100
            f.write(f"{i:<6} {lang:<20} {format_large_number(count):<15} {pct:>6.2f}%\n")
        
        f.write(f"\n")
        
        # Status Distribution
        f.write("STATUS DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Status':<30} {'Count':<15} {'Percentage':<10}\n")
        f.write("-" * 80 + "\n")
        
        for status, count in results['status_distribution']:
            pct = (count / results['total_submissions']) * 100
            f.write(f"{status:<30} {format_large_number(count):<15} {pct:>6.2f}%\n")
        
        f.write(f"\n")
        
        # Top Problems by Python Submissions
        f.write("TOP 20 PROBLEMS BY PYTHON SUBMISSIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Problem ID':<12} {'Python':<10} {'Java':<10} {'Pairs':<12}\n")
        f.write("-" * 80 + "\n")
        
        sorted_by_python = sorted(results['problem_stats'], key=lambda x: x['python'], reverse=True)[:20]
        for i, prob in enumerate(sorted_by_python, 1):
            pairs = prob['python'] * prob['java']
            f.write(f"{i:<6} {prob['problem_id']:<12} {prob['python']:<10} {prob['java']:<10} {format_large_number(pairs):<12}\n")
        
        f.write(f"\n")
        
        # Top Problems by Java Submissions
        f.write("TOP 20 PROBLEMS BY JAVA SUBMISSIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Problem ID':<12} {'Python':<10} {'Java':<10} {'Pairs':<12}\n")
        f.write("-" * 80 + "\n")
        
        sorted_by_java = sorted(results['problem_stats'], key=lambda x: x['java'], reverse=True)[:20]
        for i, prob in enumerate(sorted_by_java, 1):
            pairs = prob['python'] * prob['java']
            f.write(f"{i:<6} {prob['problem_id']:<12} {prob['python']:<10} {prob['java']:<10} {format_large_number(pairs):<12}\n")
        
        f.write(f"\n")
        
        # Top Problems by Cartesian Product
        f.write("TOP 20 PROBLEMS BY CARTESIAN PRODUCT (Python × Java pairs)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Problem ID':<12} {'Python':<10} {'Java':<10} {'Pairs':<12}\n")
        f.write("-" * 80 + "\n")
        
        # Calculate pairs and sort
        for prob in results['problem_stats']:
            prob['pairs'] = prob['python'] * prob['java']
        
        sorted_by_pairs = sorted(results['problem_stats'], key=lambda x: x['pairs'], reverse=True)[:20]
        for i, prob in enumerate(sorted_by_pairs, 1):
            f.write(f"{i:<6} {prob['problem_id']:<12} {prob['python']:<10} {prob['java']:<10} {format_large_number(prob['pairs']):<12}\n")
        
        f.write(f"\n")
        
        # Summary Statistics
        f.write("SUMMARY FOR CARTESIAN PRODUCT PAIRING\n")
        f.write("-" * 80 + "\n")
        f.write(f"If you create pairs of (Python submission, Java submission) for the same\n")
        f.write(f"problem, you will generate:\n")
        f.write(f"\n")
        f.write(f"  Total Pairs: {format_large_number(results['cartesian_product_within_problems'])}\n")
        f.write(f"\n")
        f.write(f"This assumes each Python submission is paired with every Java submission\n")
        f.write(f"within the same problem.\n")
        f.write(f"\n")
        f.write(f"Average pairs per problem (with both languages): {results['cartesian_product_within_problems'] / results['problems_with_both']:,.1f}\n")
        f.write(f"\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nSummary report written to: {output_file}")

if __name__ == "__main__":
    print("Starting metadata analysis...\n")
    
    results = analyze_metadata()
    
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print(f"Total Submissions:        {format_large_number(results['total_submissions'])}")
    print(f"Python Submissions:       {format_large_number(results['python_submissions'])}")
    print(f"Java Submissions:         {format_large_number(results['java_submissions'])}")
    print(f"Problems with Both:       {format_large_number(results['problems_with_both'])}")
    print(f"\nCartesian Product (within problems):")
    print(f"  {format_large_number(results['cartesian_product_within_problems'])} pairs")
    print("=" * 80 + "\n")
    
    output_path = '/Users/karanvirkhanna/BigCodeNet/metadataSummary.txt'
    write_summary_report(results, output_path)
    
    # Save detailed results as JSON for further analysis
    json_path = '/Users/karanvirkhanna/BigCodeNet/metadata_analysis.json'
    
    # Prepare JSON-serializable version
    json_results = {
        'total_problems': results['total_problems'],
        'total_submissions': results['total_submissions'],
        'python_submissions': results['python_submissions'],
        'java_submissions': results['java_submissions'],
        'python_accepted': results['python_accepted'],
        'java_accepted': results['java_accepted'],
        'python_acceptance_rate': results['python_acceptance_rate'],
        'java_acceptance_rate': results['java_acceptance_rate'],
        'problems_with_python': results['problems_with_python'],
        'problems_with_java': results['problems_with_java'],
        'problems_with_both': results['problems_with_both'],
        'cartesian_product_all': results['cartesian_product_all'],
        'cartesian_product_within_problems': results['cartesian_product_within_problems'],
        'language_distribution': dict(results['language_distribution']),
        'status_distribution': dict(results['status_distribution'])
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Detailed results saved to: {json_path}")
    print("\nAnalysis complete!")

