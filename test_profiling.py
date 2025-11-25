# Quick test ProfileReport
import pandas as pd
from ydata_profiling import ProfileReport

print("Loading train data...")
df = pd.read_csv("data/train.csv")

print(f"Generating EDA report for {df.shape[0]} rows × {df.shape[1]} columns...")
print("This will take 2-5 minutes...")

profile = ProfileReport(
    df, 
    title="Student Performance - Quick EDA",
    minimal=True  # Quick version
)

profile.to_file("eda_report_quick.html")
print("\n✓ Report generated: eda_report_quick.html")
print("  Open in browser to view!")
