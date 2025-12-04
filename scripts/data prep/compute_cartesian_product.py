#!/usr/bin/env python3
"""
Compute Cartesian product of Python and Java submissions
that solve the SAME problem.
"""

import csv
from pathlib import Path
from collections import defaultdict

def compute_same_problem_cartesian_product(base_path='/Users/karanvirkhanna/BigCodeNet'):
    """
    Compute the Cartesian product of Python × Java submissions
    where both submissions solve the same problem.
    """
    
    metadata_dir = Path(base_path) / 'metadata'
    problem_csvs = sorted(metadata_dir.glob('p?????.csv'))
    
    print(f"Computing Cartesian product for {len(problem_csvs)} problems...")
    print("=" * 80)
    
    # Statistics
    total_cartesian_product = 0
    problems_processed = 0
    problems_with_both = 0
    
    python_total = 0
    java_total = 0
    
    problem_details = []
    
    # Process each problem
    for i, csv_file in enumerate(problem_csvs):
        problem_id = csv_file.stem
        
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(problem_csvs)} problems...")
        
        try:
            # Read submissions for this problem
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                submissions = list(reader)
            
            # Count Python and Java submissions for THIS problem
            python_count = sum(1 for s in submissions if s['language'] == 'Python')
            java_count = sum(1 for s in submissions if s['language'] == 'Java')
            
            # Track totals
            python_total += python_count
            java_total += java_count
            problems_processed += 1
            
            # Compute Cartesian product ONLY if both languages exist
            if python_count > 0 and java_count > 0:
                pairs_for_this_problem = python_count * java_count
                total_cartesian_product += pairs_for_this_problem
                problems_with_both += 1
                
                problem_details.append({
                    'problem_id': problem_id,
                    'python_count': python_count,
                    'java_count': java_count,
                    'pairs': pairs_for_this_problem
                })
        
        except Exception as e:
            print(f"Error processing {problem_id}: {e}")
            continue
    
    print(f"Completed processing all {problems_processed} problems!")
    print("=" * 80)
    print()
    
    # Results
    print("CARTESIAN PRODUCT CALCULATION RESULTS")
    print("=" * 80)
    print()
    print("METHODOLOGY:")
    print("  For each problem that has BOTH Python AND Java submissions:")
    print("    - Count Python submissions (P)")
    print("    - Count Java submissions (J)")
    print("    - Compute pairs for this problem: P × J")
    print("  Sum all pairs across all problems")
    print()
    print("=" * 80)
    print()
    
    print("RESULTS:")
    print(f"  Total problems processed:               {problems_processed:,}")
    print(f"  Problems with Python submissions:       {sum(1 for p in problem_details if p['python_count'] > 0) + (problems_processed - problems_with_both):,}")
    print(f"  Problems with Java submissions:         {sum(1 for p in problem_details if p['java_count'] > 0) + (problems_processed - problems_with_both):,}")
    print(f"  Problems with BOTH Python AND Java:     {problems_with_both:,}")
    print()
    print(f"  Total Python submissions:               {python_total:,}")
    print(f"  Total Java submissions:                 {java_total:,}")
    print()
    print("=" * 80)
    print()
    print("🎯 CARTESIAN PRODUCT (Same-Problem Pairs):")
    print(f"    {total_cartesian_product:,} pairs")
    print()
    print("=" * 80)
    print()
    
    # Statistics
    if problems_with_both > 0:
        pairs_list = [p['pairs'] for p in problem_details]
        pairs_list.sort()
        
        mean_pairs = total_cartesian_product / problems_with_both
        median_pairs = pairs_list[len(pairs_list) // 2]
        min_pairs = pairs_list[0]
        max_pairs = pairs_list[-1]
        
        print("DISTRIBUTION STATISTICS:")
        print(f"  Mean pairs per problem:     {mean_pairs:,.1f}")
        print(f"  Median pairs per problem:   {median_pairs:,}")
        print(f"  Min pairs:                  {min_pairs:,}")
        print(f"  Max pairs:                  {max_pairs:,}")
        print()
        
        # Size categories
        small = sum(1 for p in pairs_list if p < 10000)
        medium = sum(1 for p in pairs_list if 10000 <= p < 100000)
        large = sum(1 for p in pairs_list if 100000 <= p < 1000000)
        xlarge = sum(1 for p in pairs_list if p >= 1000000)
        
        print("SIZE DISTRIBUTION:")
        print(f"  < 10K pairs:      {small:5,} problems ({small/problems_with_both*100:5.1f}%)")
        print(f"  10K - 100K:       {medium:5,} problems ({medium/problems_with_both*100:5.1f}%)")
        print(f"  100K - 1M:        {large:5,} problems ({large/problems_with_both*100:5.1f}%)")
        print(f"  >= 1M pairs:      {xlarge:5,} problems ({xlarge/problems_with_both*100:5.1f}%)")
        print()
    
    print("=" * 80)
    print()
    
    # Top 10 problems by pairs
    print("TOP 10 PROBLEMS BY NUMBER OF PAIRS:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Problem':<10} {'Python':<12} {'Java':<12} {'Pairs':<15}")
    print("-" * 80)
    
    problem_details.sort(key=lambda x: x['pairs'], reverse=True)
    for i, problem in enumerate(problem_details[:10], 1):
        print(f"{i:<6} {problem['problem_id']:<10} {problem['python_count']:<12,} "
              f"{problem['java_count']:<12,} {problem['pairs']:<15,}")
    
    print()
    print("=" * 80)
    print()
    
    # Verification
    print("VERIFICATION:")
    print(f"  Sum of all pairs = {total_cartesian_product:,}")
    print()
    
    # Show example calculation for top problem
    if problem_details:
        top_problem = problem_details[0]
        print(f"  Example (top problem {top_problem['problem_id']}):")
        print(f"    Python submissions: {top_problem['python_count']:,}")
        print(f"    Java submissions:   {top_problem['java_count']:,}")
        print(f"    Pairs: {top_problem['python_count']:,} × {top_problem['java_count']:,} = {top_problem['pairs']:,}")
    
    print()
    print("=" * 80)
    print()
    print("✅ FINAL ANSWER:")
    print(f"   When pairing Python and Java submissions that solve the SAME problem:")
    print(f"   Total pairs = {total_cartesian_product:,}")
    print()
    print("=" * 80)
    
    return {
        'total_pairs': total_cartesian_product,
        'problems_with_both': problems_with_both,
        'python_total': python_total,
        'java_total': java_total,
        'problem_details': problem_details
    }

if __name__ == "__main__":
    result = compute_same_problem_cartesian_product()

