#!/usr/bin/env python3
"""
CSV Comparison Tool
Compare two CSV submission files and show differences
Usage: python compare.py file1.csv file2.csv
"""

import pandas as pd
import sys
from pathlib import Path

def compare_csv_files(file1_path, file2_path):
    """Compare two CSV files and show detailed differences"""

    print("="*70)
    print("CSV COMPARISON TOOL")
    print("="*70)

    # Load CSV files
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        print(f"\n✓ Loaded: {file1_path}")
        print(f"✓ Loaded: {file2_path}")
    except Exception as e:
        print(f"\n✗ Error loading files: {e}")
        return

    # Basic info
    print(f"\n{'-'*70}")
    print("BASIC INFO")
    print(f"{'-'*70}")
    print(f"File 1: {file1_path.name}")
    print(f"  Shape: {df1.shape}")
    print(f"  Columns: {list(df1.columns)}")

    print(f"\nFile 2: {file2_path.name}")
    print(f"  Shape: {df2.shape}")
    print(f"  Columns: {list(df2.columns)}")

    # Check if shapes match
    if df1.shape != df2.shape:
        print(f"\n⚠ WARNING: Shapes don't match!")
        return

    # Check if columns match
    if list(df1.columns) != list(df2.columns):
        print(f"\n⚠ WARNING: Columns don't match!")
        return

    # Compare values
    print(f"\n{'-'*70}")
    print("COMPARISON RESULTS")
    print(f"{'-'*70}")

    # Assuming columns are: Student_ID, Target
    if 'Student_ID' in df1.columns and 'Target' in df1.columns:
        # Merge on Student_ID
        merged = df1.merge(df2, on='Student_ID', suffixes=('_file1', '_file2'))

        # Find differences
        different_mask = merged['Target_file1'] != merged['Target_file2']
        differences = merged[different_mask]

        total_rows = len(merged)
        num_differences = len(differences)
        num_same = total_rows - num_differences

        print(f"Total predictions: {total_rows}")
        print(f"Same predictions: {num_same} ({num_same/total_rows*100:.2f}%)")
        print(f"Different predictions: {num_differences} ({num_differences/total_rows*100:.2f}%)")

        if num_differences > 0:
            print(f"\n{'-'*70}")
            print("DIFFERENCES DETAIL")
            print(f"{'-'*70}")
            print(f"{'Student_ID':<15} {'File 1':<15} {'File 2':<15}")
            print(f"{'-'*70}")

            # Show first 20 differences
            for idx, row in differences.head(20).iterrows():
                print(f"{row['Student_ID']:<15} {row['Target_file1']:<15} {row['Target_file2']:<15}")

            if num_differences > 20:
                print(f"\n... and {num_differences - 20} more differences")

        # Class distribution comparison
        print(f"\n{'-'*70}")
        print("CLASS DISTRIBUTION")
        print(f"{'-'*70}")

        dist1 = df1['Target'].value_counts().sort_index()
        dist2 = df2['Target'].value_counts().sort_index()

        print(f"{'Class':<15} {'File 1':<15} {'File 2':<15} {'Difference':<15}")
        print(f"{'-'*70}")

        all_classes = sorted(set(dist1.index) | set(dist2.index))
        for cls in all_classes:
            count1 = dist1.get(cls, 0)
            count2 = dist2.get(cls, 0)
            diff = count2 - count1
            diff_str = f"{diff:+d}" if diff != 0 else "0"
            print(f"{cls:<15} {count1:<15} {count2:<15} {diff_str:<15}")

        # Percentage distribution
        print(f"\n{'-'*70}")
        print("PERCENTAGE DISTRIBUTION")
        print(f"{'-'*70}")
        print(f"{'Class':<15} {'File 1':<15} {'File 2':<15}")
        print(f"{'-'*70}")

        for cls in all_classes:
            pct1 = (dist1.get(cls, 0) / total_rows * 100)
            pct2 = (dist2.get(cls, 0) / total_rows * 100)
            print(f"{cls:<15} {pct1:<15.2f}% {pct2:<15.2f}%")

    else:
        print("✗ Expected columns 'Student_ID' and 'Target' not found!")

    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}\n")


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare.py <file1.csv> <file2.csv>")
        print("\nExample:")
        print("  python compare.py submission_lr_old.csv submission_lr.csv")
        print("  python compare.py submission_dt.csv submission_svm.csv")
        sys.exit(1)

    file1 = Path(sys.argv[1])
    file2 = Path(sys.argv[2])

    if not file1.exists():
        print(f"✗ Error: File not found: {file1}")
        sys.exit(1)

    if not file2.exists():
        print(f"✗ Error: File not found: {file2}")
        sys.exit(1)

    compare_csv_files(file1, file2)


if __name__ == "__main__":
    main()
