import pandas as pd
import sys
import os
import glob
from collections import Counter

def compare_csv_files(file1_path, file2_path):
    """
    Compare two CSV files and display their differences.
    
    Args:
        file1_path: Path to the first CSV file
        file2_path: Path to the second CSV file
    """
    # Read both CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    print(f"\n{'='*80}")
    print(f"Comparing CSV Files")
    print(f"{'='*80}")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print(f"{'='*80}\n")
    
    # Display basic info
    print("File 1 Shape:", df1.shape)
    print("File 2 Shape:", df2.shape)
    print()
    
    # Count Target values in each file
    print(f"\n{'='*80}")
    print("Target Value Counts")
    print(f"{'='*80}\n")
    
    print("File 1 Target Counts:")
    target_counts1 = df1['Target'].value_counts().sort_index()
    for target, count in target_counts1.items():
        percentage = (count / len(df1)) * 100
        print(f"  {target}: {count} ({percentage:.2f}%)")
    print(f"  Total: {len(df1)}")
    
    print("\nFile 2 Target Counts:")
    target_counts2 = df2['Target'].value_counts().sort_index()
    for target, count in target_counts2.items():
        percentage = (count / len(df2)) * 100
        print(f"  {target}: {count} ({percentage:.2f}%)")
    print(f"  Total: {len(df2)}")
    
    # Compare the two files
    print(f"\n{'='*80}")
    print("Differences Analysis")
    print(f"{'='*80}\n")
    
    # Check if they have the same Student_IDs
    if 'Student_ID' in df1.columns and 'Student_ID' in df2.columns:
        # Merge on Student_ID
        merged = pd.merge(df1, df2, on='Student_ID', suffixes=('_file1', '_file2'), how='outer', indicator=True)
        
        # Find rows only in file1
        only_in_file1 = merged[merged['_merge'] == 'left_only']
        print(f"Rows only in File 1: {len(only_in_file1)}")
        
        # Find rows only in file2
        only_in_file2 = merged[merged['_merge'] == 'right_only']
        print(f"Rows only in File 2: {len(only_in_file2)}")
        
        # Find rows in both files
        in_both = merged[merged['_merge'] == 'both']
        print(f"Rows in both files: {len(in_both)}")
        
        # Find differences in Target values for common Student_IDs
        if 'Target_file1' in in_both.columns and 'Target_file2' in in_both.columns:
            differences = in_both[in_both['Target_file1'] != in_both['Target_file2']]
            print(f"\nRows with different Target values: {len(differences)}")
            
            if len(differences) > 0:
                print("\nTarget Value Changes:")
                change_summary = differences.groupby(['Target_file1', 'Target_file2']).size().reset_index(name='count')
                for _, row in change_summary.iterrows():
                    print(f"  {row['Target_file1']} -> {row['Target_file2']}: {row['count']} changes")
                
                print("\nFirst 20 differences:")
                print(differences[['Student_ID', 'Target_file1', 'Target_file2']].head(20).to_string(index=False))
                
                if len(differences) > 20:
                    print(f"\n... and {len(differences) - 20} more differences")
        
        # Agreement percentage
        if len(in_both) > 0:
            agreement = len(in_both[in_both['Target_file1'] == in_both['Target_file2']])
            agreement_pct = (agreement / len(in_both)) * 100
            print(f"\nAgreement: {agreement}/{len(in_both)} ({agreement_pct:.2f}%)")
    else:
        print("Cannot compare: 'Student_ID' column not found in one or both files")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    # Hardcoded reference file
    reference_file = "../submission/submission_logreg_20251202_222512.csv"
    
    # Get the latest file from submission folder
    submission_folder = "../submission"
    csv_files = glob.glob(os.path.join(submission_folder, "*.csv"))
    
    if not csv_files:
        print("Error: No CSV files found in submission folder")
        sys.exit(1)
    
    # Sort by modification time and get the latest
    latest_file = max(csv_files, key=os.path.getmtime)
    
    print(f"Reference file: {reference_file}")
    print(f"Latest file: {latest_file}")
    print(f"Latest file modified: {pd.Timestamp.fromtimestamp(os.path.getmtime(latest_file))}")
    
    try:
        compare_csv_files(reference_file, latest_file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
