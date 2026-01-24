"""Example analysis script for evaluation results."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.result_saver import load_results


def analyze_results(results_dir: str):
    """
    Analyze evaluation results and generate summary.
    
    Args:
        results_dir: Directory containing results
    """
    # Load results
    data = load_results(results_dir)
    results = data.get('results', {})
    metrics = data.get('metrics', {})
    
    print("=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nExperiment: {data.get('metadata', {}).get('experiment_name', 'unknown')}")
    print(f"Model: {data.get('metadata', {}).get('model', 'unknown')}")
    print(f"Examples: {data.get('metadata', {}).get('num_examples', 0)}")
    
    print("\n" + "-" * 60)
    print("METRICS")
    print("-" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument('results_dir', help="Directory containing results")
    args = parser.parse_args()
    
    analyze_results(args.results_dir)


if __name__ == '__main__':
    main()
