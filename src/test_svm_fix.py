"""
Quick test to verify SVM improvements
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from svm import SVM_OneVsRest, SVM_OneVsOne, SVM_DAGSVM

# Set random seed
np.random.seed(67)

# Load data
print("Loading data...")
train_df = pd.read_csv('data/train.csv')

# Prepare features and target
X = train_df.drop(columns=['Target', 'Student_ID'])
y = train_df['Target']

# Take a smaller sample for quick testing
print("Sampling 1000 rows for quick test...")
sample_indices = np.random.choice(len(X), 1000, replace=False)
X_sample = X.iloc[sample_indices]
y_sample = y.iloc[sample_indices]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=67, stratify=y_sample
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Classes: {np.unique(y_train)}")

# CRITICAL: Scale the features!
print("\n" + "="*60)
print("SCALING FEATURES (CRITICAL FOR SVM!)")
print("="*60)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(f"Feature scale before: min={X_train.min().min():.2f}, max={X_train.max().max():.2f}")
print(f"Feature scale after:  min={X_train_scaled.min().min():.2f}, max={X_train_scaled.max().max():.2f}")

# Test with improved parameters
print("\n" + "="*60)
print("TESTING SVM WITH IMPROVED PARAMETERS")
print("="*60)
print("New parameters:")
print("  - Learning rate: 0.01 (was 0.001)")
print("  - Iterations: 5000 (was 1000)")
print("  - Lambda param: 0.01 (was 1/n_iters)")
print("  - Features: SCALED (was unscaled)")

# Test DAGSVM
print("\n[1] Testing SVM DAGSVM...")
svm_dag = SVM_DAGSVM(lr=0.01, n_iters=2000, C=1.0, lambda_param=0.01)
svm_dag.fit(X_train_scaled, y_train)

y_pred_dag = svm_dag.predict(X_test_scaled)
acc_dag = accuracy_score(y_test, y_pred_dag)
print(f"DAGSVM Accuracy: {acc_dag:.4f}")

# Test One-vs-Rest
print("\n[2] Testing SVM One-vs-Rest...")
svm_ovr = SVM_OneVsRest(lr=0.01, n_iters=2000, C=1.0, lambda_param=0.01)
svm_ovr.fit(X_train_scaled, y_train)

y_pred_ovr = svm_ovr.predict(X_test_scaled)
acc_ovr = accuracy_score(y_test, y_pred_ovr)
print(f"One-vs-Rest Accuracy: {acc_ovr:.4f}")

# Test One-vs-One
print("\n[3] Testing SVM One-vs-One...")
svm_ovo = SVM_OneVsOne(lr=0.01, n_iters=2000, C=1.0, lambda_param=0.01)
svm_ovo.fit(X_train_scaled, y_train)

y_pred_ovo = svm_ovo.predict(X_test_scaled)
acc_ovo = accuracy_score(y_test, y_pred_ovo)
print(f"One-vs-One Accuracy: {acc_ovo:.4f}")

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"DAGSVM:      {acc_dag:.4f}")
print(f"One-vs-Rest: {acc_ovr:.4f}")
print(f"One-vs-One:  {acc_ovo:.4f}")
print("\nNote: These results are on a small sample (1000 rows)")
print("      Full training will take longer but should be more accurate")
print("="*60)
