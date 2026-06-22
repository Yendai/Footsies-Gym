#!/usr/bin/env python3
"""
Validation script for entropy tracking in train_ppo.py

This script checks if the entropy CSV output shows signs of proper policy learning.
It validates fixes for the 6 critical entropy tracking errors.

USAGE:
    python validate_entropy.py [--csv-path PATH] [--verbose]

EXPECTED OUTPUT FOR WORKING SYSTEM:
    - Entropy starts near ln(16) ≈ 2.7726 (random policy)
    - Entropy decreases over time as policy focuses on good actions
    - Final entropy significantly below maximum (< 1.0-2.0 for converged policy)
    - Multiple policy updates logged (update_count > 1)

EXPECTED OUTPUT FOR BROKEN SYSTEM:
    - Entropy stays constant at ~2.7726 across all updates
    - No variance in entropy values
    - Single or very few policy updates
"""

import argparse
import csv
import math
import sys
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "footsies_gym"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def load_entropy_csv(csv_path: str) -> list:
    """Load entropy tracking CSV file."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows
    except FileNotFoundError:
        print(f"ERROR: CSV file not found: {csv_path}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        return None


def analyze_entropy_data(rows: list) -> dict:
    """Analyze entropy data for signs of proper learning."""
    if not rows:
        return {"error": "No data rows found in CSV"}

    required_columns = [
        "timesteps",
        "update_count",
        "mean_entropy",
        "std_entropy",
        "min_entropy",
        "max_entropy",
    ]

    missing_columns = [column for column in required_columns if column not in rows[0]]
    if missing_columns:
        return {"error": f"CSV missing columns: {', '.join(missing_columns)}"}

    try:
        data = {
            "timesteps": [int(row["timesteps"]) for row in rows],
            "update_count": [int(row["update_count"]) for row in rows],
            "mean_entropy": [float(row["mean_entropy"]) for row in rows],
            "std_entropy": [float(row["std_entropy"]) for row in rows],
            "min_entropy": [float(row["min_entropy"]) for row in rows],
            "max_entropy": [float(row["max_entropy"]) for row in rows],
        }
    except (KeyError, TypeError, ValueError) as e:
        return {"error": f"CSV format error: {e}"}
    
    max_theoretical_entropy = math.log(16)  # ln(16) for Discrete(16)
    first_entropy = data['mean_entropy'][0]
    last_entropy = data['mean_entropy'][-1]
    entropy_change = first_entropy - last_entropy
    
    analysis = {
        "num_updates": len(rows),
        "timesteps_total": data['timesteps'][-1],
        "entropy_start": first_entropy,
        "entropy_end": last_entropy,
        "entropy_change": entropy_change,
        "entropy_change_percent": (entropy_change / first_entropy * 100) if first_entropy > 0 else 0,
        "entropy_std": sum(data['std_entropy']) / len(data['std_entropy']),
        "max_theoretical": max_theoretical_entropy,
        "entropy_relative_to_max": last_entropy / max_theoretical_entropy,
    }
    
    return analysis


def validate_entropy_tracking(analysis: dict, verbose: bool = False) -> dict:
    """Validate that entropy tracking shows proper learning."""
    if 'error' in analysis:
        return analysis
    
    max_entropy = analysis['max_theoretical']
    results = {
        "checks": {},
        "warnings": [],
        "errors": [],
        "info": []
    }
    
    # CHECK 1: Is entropy decreasing? (Sign of learning)
    entropy_decreasing = analysis['entropy_end'] < analysis['entropy_start']
    results['checks']['entropy_decreasing'] = {
        'passed': entropy_decreasing,
        'description': 'Entropy should decrease as policy learns',
        'value': f"{analysis['entropy_start']:.4f} → {analysis['entropy_end']:.4f}",
    }
    if not entropy_decreasing:
        results['errors'].append("FAILED: Entropy not decreasing. Policy may not be learning.")
    
    # CHECK 2: Is final entropy below maximum? (Sign of policy focus)
    entropy_away_from_max = analysis['entropy_end'] < max_entropy * 0.95
    results['checks']['entropy_focused'] = {
        'passed': entropy_away_from_max,
        'description': 'Final entropy should be significantly below max (>5% change)',
        'value': f"{analysis['entropy_end']:.4f} vs max {max_entropy:.4f}",
    }
    if not entropy_away_from_max:
        results['warnings'].append(f"WARNING: Entropy is very close to max ({analysis['entropy_relative_to_max']*100:.1f}%). "
                                   "Policy may not have learned much yet, or entropy tracking is broken.")
    
    # CHECK 3: Is there significant change? (>10% decay)
    significant_change = analysis['entropy_change_percent'] > 10
    results['checks']['significant_change'] = {
        'passed': significant_change,
        'description': 'Entropy should decay by >10% as policy learns',
        'value': f"{analysis['entropy_change_percent']:.1f}% decay",
    }
    if not significant_change:
        results['errors'].append(f"FAILED: Entropy change too small ({analysis['entropy_change_percent']:.1f}%). "
                                "Error 4 may still be present (constant maximum entropy).")
    
    # CHECK 4: Multiple policy updates? (Sign of Error 5 being fixed)
    multiple_updates = analysis['num_updates'] > 1
    results['checks']['multiple_updates'] = {
        'passed': multiple_updates,
        'description': 'Should have multiple policy updates logged',
        'value': f"{analysis['num_updates']} updates",
    }
    if not multiple_updates:
        results['errors'].append("FAILED: Only 1 update logged. Callback may not be working.")
    
    # CHECK 5: Reasonable entropy variance (not flat)
    entropy_flat = analysis['entropy_std'] < 0.01
    results['checks']['entropy_variance'] = {
        'passed': not entropy_flat,
        'description': 'Entropy should vary across updates (not constant)',
        'value': f"std={analysis['entropy_std']:.6f}",
    }
    if entropy_flat:
        results['warnings'].append("WARNING: Entropy is nearly constant across updates. "
                                   "May indicate Error 5 (per-timestep logging) or Error 1 (measuring before training).")
    
    # DIAGNOSTIC INFO
    results['info'].append(f"Total timesteps: {analysis['timesteps_total']}")
    results['info'].append(f"Policy updates captured: {analysis['num_updates']}")
    results['info'].append(f"Entropy decay: {analysis['entropy_change']:.4f} ({analysis['entropy_change_percent']:.1f}%)")
    results['info'].append(f"Avg entropy std per update: {analysis['entropy_std']:.6f}")
    
    return results


def print_results(analysis: dict, validation: dict, verbose: bool = False):
    """Print validation results in human-readable format."""
    print("\n" + "="*70)
    print("ENTROPY TRACKING VALIDATION REPORT")
    print("="*70)
    
    # Analysis section
    print("\nDATA SUMMARY:")
    print(f"  Total updates logged:        {analysis.get('num_updates', 'N/A')}")
    print(f"  Total timesteps collected:   {analysis.get('timesteps_total', 'N/A')}")
    print(f"  Entropy at start:            {analysis.get('entropy_start', float('nan')):.6f}")
    print(f"  Entropy at end:              {analysis.get('entropy_end', float('nan')):.6f}")
    print(f"  Entropy change:              {analysis.get('entropy_change', float('nan')):.6f} ({analysis.get('entropy_change_percent', float('nan')):.1f}%)")
    print(f"  Max theoretical entropy:     {analysis.get('max_theoretical', float('nan')):.6f} (ln(16))")
    print(f"  Final entropy vs max:        {analysis.get('entropy_relative_to_max', float('nan'))*100:.1f}%")
    
    # Validation checks
    print("\nVALIDATION CHECKS:")
    for check_name, check_result in validation.get('checks', {}).items():
        status = "✓ PASS" if check_result['passed'] else "✗ FAIL"
        print(f"  {status} | {check_result['description']}")
        print(f"         {check_result['value']}")
    
    # Errors
    if validation.get('errors'):
        print("\nERRORS:")
        for error in validation['errors']:
            print(f"  ✗ {error}")
    
    # Warnings
    if validation.get('warnings'):
        print("\nWARNINGS:")
        for warning in validation['warnings']:
            print(f"  ⚠ {warning}")
    
    # Info
    if validation.get('info'):
        print("\nINFORMATION:")
        for info in validation['info']:
            print(f"  ℹ {info}")
    
    # Summary
    print("\nSUMMARY:")
    if not validation.get('errors') and not validation.get('warnings'):
        print("  ✓ Entropy tracking appears to be working correctly!")
        print("    Policy is learning and entropy shows expected decay pattern.")
    elif not validation.get('errors') and validation.get('warnings'):
        print("  ⚠ Entropy tracking is partially working.")
        print("    Policy may need more training, or there might be minor issues.")
    else:
        print("  ✗ Entropy tracking is broken!")
        print("    One or more critical checks failed. Review the errors above.")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate entropy tracking CSV from train_ppo.py",
        epilog="See script header comments for details on expected output."
    )
    parser.add_argument("--csv-path", default="Agents-Test/policy_entropy.csv",
                       help="Path to policy entropy CSV file")
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose debug information")
    args = parser.parse_args()
    
    # Load CSV
    rows = load_entropy_csv(args.csv_path)
    if rows is None:
        return 1
    
    # Analyze data
    analysis = analyze_entropy_data(rows)
    if 'error' in analysis:
        print(f"ERROR: {analysis['error']}")
        return 1
    
    # Validate
    validation = validate_entropy_tracking(analysis, verbose=args.verbose)
    
    # Print results
    print_results(analysis, validation, verbose=args.verbose)
    
    # Return exit code based on errors
    if validation.get('errors'):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
