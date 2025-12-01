

# ============================================================
# IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# IMPORT DATASET
# ============================================================
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nFirst few rows:")
print(train_df.head())

# ============================================================
# DEFINE CATEGORICAL AND CONTINUOUS FEATURES
# ============================================================
categorical_features = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification', 'Nationality',
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
]

continuous_features = [
    'Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

target_column = 'Target'

print("\n" + "="*60)
print("FEATURE TYPES")
print("="*60)
print(f"Categorical features: {len(categorical_features)}")
print(f"Continuous features: {len(continuous_features)}")
print(f"Total features: {len(categorical_features) + len(continuous_features)}")

# ============================================================
# SPLIT TRAINING DATA (TRAIN + VALIDATION)
# ============================================================
X = train_df.drop(columns=[target_column])
y = train_df[target_column]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "="*60)
print("DATA SPLIT")
print("="*60)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {test_df.shape[0]} samples")

# ============================================================
# DATA CLEANING - MISSING VALUES
# ============================================================
print("\n" + "="*60)
print("MISSING VALUES CHECK")
print("="*60)
missing_train = X_train.isnull().sum()
missing_train = missing_train[missing_train > 0]

if len(missing_train) > 0:
    print("Missing values in training data:")
    print(missing_train)
else:
    print("No missing values in training data")

# ============================================================
# DATA CLEANING - DUPLICATES
# ============================================================
print("\n" + "="*60)
print("DUPLICATE CHECK")
print("="*60)
duplicates = X_train.duplicated().sum()
print(f"Duplicate rows in training data: {duplicates}")

if duplicates > 0:
    X_train = X_train.drop_duplicates()
    y_train = y_train[X_train.index]
    print(f"After removing duplicates: {X_train.shape[0]} samples")

# ============================================================
# DEALING WITH OUTLIERS - CONTINUOUS FEATURES
# ============================================================
print("\n" + "="*60)
print("OUTLIER DETECTION AND HANDLING")
print("="*60)

X_train_clean = X_train.copy()
X_val_clean = X_val.copy()

outlier_info = {}

# IQR method with adaptive factors
for col in continuous_features:
    if col not in X_train_clean.columns:
        continue
    
    Q1 = X_train_clean[col].quantile(0.25)
    Q3 = X_train_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Adaptive factor based on feature type
    if 'age' in col.lower():
        factor = 4.0  # Most loose for age
    elif 'curricular' in col.lower():
        factor = 4.0  # Loose for curricular (grades can be 0)
    elif 'credited' in col.lower() or 'without evaluation' in col.lower():
        continue  # Skip these columns (mostly zeros)
    else:
        factor = 2.5  # Default factor
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # For curricular columns, ensure lower bound is 0 (grades can't be negative)
    if 'curricular' in col.lower():
        lower_bound = 0.0
    
    # Store bounds for test data
    outlier_info[col] = {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'factor': factor
    }
    
    # Clip outliers in training data
    outliers_train = ((X_train_clean[col] < lower_bound) | (X_train_clean[col] > upper_bound)).sum()
    X_train_clean[col] = X_train_clean[col].clip(lower_bound, upper_bound)
    
    # Clip outliers in validation data
    outliers_val = ((X_val_clean[col] < lower_bound) | (X_val_clean[col] > upper_bound)).sum()
    X_val_clean[col] = X_val_clean[col].clip(lower_bound, upper_bound)
    
    if outliers_train > 0 or outliers_val > 0:
        print(f"{col}:")
        print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}] (factor={factor})")
        print(f"  Train outliers clipped: {outliers_train}")
        print(f"  Val outliers clipped: {outliers_val}")

# ============================================================
# DEALING WITH OUTLIERS - CATEGORICAL FEATURES
# ============================================================
print("\n" + "="*60)
print("CATEGORICAL OUTLIER HANDLING")
print("="*60)

categorical_ranges = {
    'Marital status': (1, 6),
    'Application mode': (1, 53),
    'Application order': (0, 9)
}

# Calculate mode for each categorical feature from training data
categorical_modes = {}
for col in categorical_features:
    if col in X_train_clean.columns:
        categorical_modes[col] = X_train_clean[col].mode()[0]

# Handle outliers in categorical features
for col, (min_val, max_val) in categorical_ranges.items():
    if col in X_train_clean.columns:
        # Convert to numeric temporarily for comparison
        train_numeric = pd.to_numeric(X_train_clean[col], errors='coerce')
        val_numeric = pd.to_numeric(X_val_clean[col], errors='coerce')
        
        # Find outliers
        train_outliers = ((train_numeric < min_val) | (train_numeric > max_val) | train_numeric.isna()).sum()
        val_outliers = ((val_numeric < min_val) | (val_numeric > max_val) | val_numeric.isna()).sum()
        
        # Replace outliers with mode
        mode_val = categorical_modes[col]
        mask_train = (train_numeric < min_val) | (train_numeric > max_val) | train_numeric.isna()
        mask_val = (val_numeric < min_val) | (val_numeric > max_val) | val_numeric.isna()
        
        X_train_clean.loc[mask_train, col] = mode_val
        X_val_clean.loc[mask_val, col] = mode_val
        
        if train_outliers > 0 or val_outliers > 0:
            print(f"{col}: Valid range [{min_val}, {max_val}]")
            print(f"  Train outliers replaced: {train_outliers} -> mode={mode_val}")
            print(f"  Val outliers replaced: {val_outliers}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

def add_features(df):
    df_new = df.copy()
    
    # Academic performance features
    df_new['Total_Units_Approved'] = (
        df_new['Curricular units 1st sem (approved)'] + 
        df_new['Curricular units 2nd sem (approved)']
    )
    
    df_new['Total_Units_Enrolled'] = (
        df_new['Curricular units 1st sem (enrolled)'] + 
        df_new['Curricular units 2nd sem (enrolled)']
    )
    
    df_new['Approval_Rate'] = np.where(
        df_new['Total_Units_Enrolled'] > 0,
        df_new['Total_Units_Approved'] / df_new['Total_Units_Enrolled'],
        0
    )
    
    df_new['Average_Grade'] = (
        df_new['Curricular units 1st sem (grade)'] + 
        df_new['Curricular units 2nd sem (grade)']
    ) / 2
    
    df_new['Grade_Difference'] = (
        df_new['Curricular units 2nd sem (grade)'] - 
        df_new['Curricular units 1st sem (grade)']
    )
    
    return df_new

X_train_clean = add_features(X_train_clean)
X_val_clean = add_features(X_val_clean)

new_features = ['Total_Units_Approved', 'Total_Units_Enrolled', 'Approval_Rate', 
                'Average_Grade', 'Grade_Difference']
continuous_features.extend(new_features)

print(f"Added {len(new_features)} new features:")
for feat in new_features:
    print(f"  - {feat}")

# ============================================================
# FEATURE SELECTION - CHI-SQUARE FOR CATEGORICAL
# ============================================================
print("\n" + "="*60)
print("FEATURE SELECTION - CHI-SQUARE (CATEGORICAL)")
print("="*60)

# Prepare data for Chi-Square test
X_train_cat = X_train_clean[categorical_features].copy()
X_val_cat = X_val_clean[categorical_features].copy()

# Label encode target
le_target = LabelEncoder()
y_train_encoded = le_target.fit_transform(y_train)
y_val_encoded = le_target.transform(y_val)

# Label encode categorical features
le_dict_cat = {}
for col in categorical_features:
    le = LabelEncoder()
    X_train_cat[col] = le.fit_transform(X_train_cat[col].astype(str))
    X_val_cat[col] = le.transform(X_val_cat[col].astype(str))
    le_dict_cat[col] = le

# Apply Chi-Square test
k_best_cat = 15  # Select top 15 categorical features
chi2_selector = SelectKBest(chi2, k=k_best_cat)
X_train_cat_selected = chi2_selector.fit_transform(X_train_cat, y_train_encoded)
X_val_cat_selected = chi2_selector.transform(X_val_cat)

# Get selected feature names
chi2_scores = chi2_selector.scores_
chi2_selected_idx = chi2_selector.get_support(indices=True)
selected_categorical = [categorical_features[i] for i in chi2_selected_idx]

print(f"Selected {len(selected_categorical)} out of {len(categorical_features)} categorical features")
print("\nTop 10 categorical features by Chi-Square score:")
feature_scores = list(zip(categorical_features, chi2_scores))
feature_scores.sort(key=lambda x: x[1], reverse=True)
for feat, score in feature_scores[:10]:
    selected = "✓" if feat in selected_categorical else "✗"
    print(f"  {selected} {feat}: {score:.2f}")

# ============================================================
# FEATURE SELECTION - ANOVA F-TEST FOR CONTINUOUS
# ============================================================
print("\n" + "="*60)
print("FEATURE SELECTION - ANOVA F-TEST (CONTINUOUS)")
print("="*60)

X_train_cont = X_train_clean[continuous_features].copy()
X_val_cont = X_val_clean[continuous_features].copy()

# Apply ANOVA F-test
k_best_cont = 15  # Select top 15 continuous features
anova_selector = SelectKBest(f_classif, k=k_best_cont)
X_train_cont_selected = anova_selector.fit_transform(X_train_cont, y_train_encoded)
X_val_cont_selected = anova_selector.transform(X_val_cont)

# Get selected feature names
anova_scores = anova_selector.scores_
anova_selected_idx = anova_selector.get_support(indices=True)
selected_continuous = [continuous_features[i] for i in anova_selected_idx]

print(f"Selected {len(selected_continuous)} out of {len(continuous_features)} continuous features")
print("\nTop 10 continuous features by ANOVA F-score:")
feature_scores = list(zip(continuous_features, anova_scores))
feature_scores.sort(key=lambda x: x[1], reverse=True)
for feat, score in feature_scores[:10]:
    selected = "✓" if feat in selected_continuous else "✗"
    print(f"  {selected} {feat}: {score:.2f}")

# ============================================================
# COMBINE SELECTED FEATURES AND SCALE
# ============================================================
print("\n" + "="*60)
print("FINAL FEATURE SET PREPARATION")
print("="*60)

# Combine selected categorical and continuous features
X_train_final = pd.DataFrame(
    np.concatenate([X_train_cat_selected, X_train_cont_selected], axis=1),
    columns=selected_categorical + selected_continuous
)

X_val_final = pd.DataFrame(
    np.concatenate([X_val_cat_selected, X_val_cont_selected], axis=1),
    columns=selected_categorical + selected_continuous
)

print(f"Total selected features: {X_train_final.shape[1]}")
print(f"  - Categorical: {len(selected_categorical)}")
print(f"  - Continuous: {len(selected_continuous)}")

# Scale continuous features
scaler = StandardScaler()
X_train_final[selected_continuous] = scaler.fit_transform(X_train_final[selected_continuous])
X_val_final[selected_continuous] = scaler.transform(X_val_final[selected_continuous])

print("\nScaling applied to continuous features")

# ============================================================
# DECISION TREE MODEL TRAINING
# ============================================================
print("\n" + "="*60)
print("DECISION TREE TRAINING")
print("="*60)

# Train Decision Tree with CART algorithm (default in sklearn)
dt_model = DecisionTreeClassifier(
    criterion='gini',  # CART uses Gini impurity
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

dt_model.fit(X_train_final, y_train_encoded)

# Predictions
y_train_pred = dt_model.predict(X_train_final)
y_val_pred = dt_model.predict(X_val_final)

# Evaluate
train_acc = accuracy_score(y_train_encoded, y_train_pred)
val_acc = accuracy_score(y_val_encoded, y_val_pred)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

print("\nClassification Report (Validation):")
print(classification_report(y_val_encoded, y_val_pred, target_names=le_target.classes_))

print("\nConfusion Matrix (Validation):")
cm = confusion_matrix(y_val_encoded, y_val_pred)
print(cm)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE (TOP 15)")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': selected_categorical + selected_continuous,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

# ============================================================
# TEST DATA PREPROCESSING AND PREDICTION
# ============================================================
print("\n" + "="*60)
print("TEST DATA PREPROCESSING")
print("="*60)

X_test = test_df.copy()

# 1. Handle numerical outliers using training bounds
print("1. Clipping numerical outliers...")
for col in continuous_features[:len(continuous_features)-len(new_features)]:  # Original continuous features only
    if col in outlier_info and col in X_test.columns:
        lower = outlier_info[col]['lower_bound']
        upper = outlier_info[col]['upper_bound']
        outliers_count = ((X_test[col] < lower) | (X_test[col] > upper)).sum()
        X_test[col] = X_test[col].clip(lower, upper)
        if outliers_count > 0:
            print(f"   {col}: {outliers_count} outliers clipped")

# 2. Handle categorical outliers
print("\n2. Handling categorical outliers...")
for col, (min_val, max_val) in categorical_ranges.items():
    if col in X_test.columns:
        test_numeric = pd.to_numeric(X_test[col], errors='coerce')
        mode_val = categorical_modes[col]
        mask = (test_numeric < min_val) | (test_numeric > max_val) | test_numeric.isna()
        outliers_count = mask.sum()
        X_test.loc[mask, col] = mode_val
        if outliers_count > 0:
            print(f"   {col}: {outliers_count} outliers replaced with mode={mode_val}")

# 3. Feature engineering
print("\n3. Creating engineered features...")
X_test = add_features(X_test)

# 4. Encode categorical features
print("\n4. Encoding categorical features...")
X_test_cat = X_test[categorical_features].copy()
for col in categorical_features:
    X_test_cat[col] = le_dict_cat[col].transform(X_test_cat[col].astype(str))

# 5. Select features
print("\n5. Selecting features...")
X_test_cat_selected = chi2_selector.transform(X_test_cat)
X_test_cont = X_test[continuous_features].copy()
X_test_cont_selected = anova_selector.transform(X_test_cont)

X_test_final = pd.DataFrame(
    np.concatenate([X_test_cat_selected, X_test_cont_selected], axis=1),
    columns=selected_categorical + selected_continuous
)

# 6. Scale continuous features
print("\n6. Scaling continuous features...")
X_test_final[selected_continuous] = scaler.transform(X_test_final[selected_continuous])

print(f"\nTest data ready: {X_test_final.shape}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
print("\n" + "="*60)
print("GENERATING SUBMISSION FILE")
print("="*60)

y_test_pred = dt_model.predict(X_test_final)
y_test_pred_labels = le_target.inverse_transform(y_test_pred)

submission = pd.DataFrame({
    'id': test_df['Student_ID'],
    'Target': y_test_pred_labels
})

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
submission_filename = f'../submission/submission_DTL_{timestamp}.csv'
submission.to_csv(submission_filename, index=False)

print(f"Submission file saved: {submission_filename}")
print(f"Total predictions: {len(submission)}")
print("\nPrediction distribution:")
print(submission['Target'].value_counts())

print("\n" + "="*60)
print("DECISION TREE PIPELINE COMPLETED")
print("="*60)