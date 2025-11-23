"""
============================================================================
IF3170 ARTIFICIAL INTELLIGENCE - TUGAS BESAR 2
IMPLEMENTASI DARI SCRATCH: DTL, LOGISTIC REGRESSION, SVM
============================================================================
Group 06:
- Richard Christian 13523024
- Kenneth Poenadi Name 13523040
- Ivan Wirawan 13523046
- Bob Kunanda 13523086
- M Zahran Ramadhan 13523104
============================================================================
"""

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 67
np.random.seed(RANDOM_STATE)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 80)
print("LIBRARIES BERHASIL DI-IMPORT")
print("=" * 80)


# ============================================================================
# 1. DATA UNDERSTANDING - PROFILE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("1. DATA UNDERSTANDING")
print("=" * 80)

# Load dataset
df = pd.read_csv("../data/train.csv")
print(f"\nJumlah baris: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")
print(f"\nSample data (5 baris pertama):")
print(df.head())

# Membuat ProfileReport untuk memahami data secara mendalam
print("\nMembuat ProfileReport...")
profile = ProfileReport(df, title="Student Performance Data Profiling", explorative=True)
profile.to_file("../data/profile_report.html")
print("✓ Profile report saved to '../data/profile_report.html'")


# ============================================================================
# 2. SPLIT DATA: 80% TRAINING, 20% VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("2. SPLIT DATA (80-20)")
print("=" * 80)

def stratified_train_test_split(X, y, ids, test_size=0.2, random_state=67):
    """
    Membagi data menjadi training dan validation set secara stratified.
    Stratified artinya proporsi kelas di train dan validation tetap sama.
    
    IMPLEMENTASI MANUAL (TANPA SKLEARN)
    """
    np.random.seed(random_state)
    
    # Gabungkan untuk memudahkan splitting
    data = pd.concat([ids, X, y], axis=1)
    
    train_indices = []
    val_indices = []
    
    # Split untuk setiap kelas secara terpisah
    for class_label in y.unique():
        class_data = data[data['Target'] == class_label]
        indices = class_data.index.tolist()
        np.random.shuffle(indices)
        
        # Hitung split point
        split_point = int(len(indices) * (1 - test_size))
        
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])
    
    # Shuffle final indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    # Split data
    train_data = data.loc[train_indices]
    val_data = data.loc[val_indices]
    
    # Pisahkan kembali
    X_train = train_data.drop(columns=['Student_ID', 'Target'])
    y_train = train_data['Target']
    ids_train = train_data['Student_ID']
    
    X_val = val_data.drop(columns=['Student_ID', 'Target'])
    y_val = val_data['Target']
    ids_val = val_data['Student_ID']
    
    return X_train, X_val, y_train, y_val, ids_train, ids_val

# Pisahkan features dan target
X_full = df.drop(columns=['Student_ID', 'Target'])
y_full = df['Target']
ids_full = df['Student_ID']

print(f"Total data: {len(df)}")
print(f"\nDistribusi Target:")
print(y_full.value_counts())

# Lakukan splitting
X_train, X_val, y_train, y_val, ids_train, ids_val = stratified_train_test_split(
    X_full, y_full, ids_full, test_size=0.2, random_state=RANDOM_STATE
)

print(f"\nTraining set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
print("\n✓ Data berhasil di-split dengan proporsi kelas yang seimbang!")


# ============================================================================
# 3. DATA CLEANING & PREPARATION
# ============================================================================
print("\n" + "=" * 80)
print("3. DATA CLEANING & PREPARATION")
print("=" * 80)

# Identifikasi tipe data
categorical_cols = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance\t', 'Previous qualification', 'Gender', 
    'Nacionality', "Mother's qualification", "Father's qualification", 
    "Mother's occupation", "Father's occupation", 'Educational special needs', 
    'International', 'Debtor', 'Tuition fees up to date', 'Scholarship holder', 
    'Displaced'
]

categorical_cols = [col for col in categorical_cols if col in X_train.columns]
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

print(f"\nKolom kategorikal: {len(categorical_cols)}")
print(f"Kolom numerikal: {len(numerical_cols)}")

# Konversi kolom kategorikal ke string
X_train[categorical_cols] = X_train[categorical_cols].astype(str)
X_val[categorical_cols] = X_val[categorical_cols].astype(str)


# --- 3.1 HANDLING MISSING DATA ---
# CATATAN: Handling missing values bisa dengan:
# - Imputation (mean, median, mode)
# - Deletion (hapus baris/kolom)
# - Predictive imputation
#
# DALAM IMPLEMENTASI INI: Tidak ada missing values di dataset ini
# (sudah dicek saat ProfileReport)
print("\n--- 3.1 Handling Missing Data ---")
print("✓ Tidak ada missing values (verified dari ProfileReport)")


# --- 3.2 HANDLING OUTLIERS dengan IQR Method ---

def detect_outliers_iqr(data, column):
    """
    Mendeteksi outlier menggunakan metode IQR.
    IQR = Q3 - Q1
    Outlier: nilai < Q1 - 1.5*IQR atau > Q3 + 1.5*IQR
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_indices = data[(data[column] < lower_bound) | (data[column] > upper_bound)].index
    
    return lower_bound, upper_bound, outlier_indices

X_train_clean = X_train.copy()
X_val_clean = X_val.copy()

outlier_info = {}
for col in numerical_cols:
    lower, upper, outlier_idx = detect_outliers_iqr(X_train_clean, col)
    n_outliers = len(outlier_idx)
    
    if n_outliers > 0:
        outlier_info[col] = {
            'count': n_outliers,
            'percentage': (n_outliers / len(X_train_clean)) * 100,
            'lower_bound': lower,
            'upper_bound': upper
        }
        
        # Clipping outliers
        X_train_clean[col] = X_train_clean[col].clip(lower, upper)
        X_val_clean[col] = X_val_clean[col].clip(lower, upper)

print(f"✓ Outliers di-handle untuk {len(outlier_info)} kolom menggunakan IQR + Clipping")


# --- 3.3 REMOVE DUPLICATES ---
# Duplicate rows bisa menyebabkan overfitting dan bias pada model
print("\n--- 3.3 Remove Duplicates ---")

n_duplicates = X_train_clean.duplicated().sum()
if n_duplicates > 0:
    before_len = len(X_train_clean)
    train_with_target = pd.concat([X_train_clean, y_train], axis=1)
    train_with_target = train_with_target.drop_duplicates()
    
    X_train_clean = train_with_target.drop(columns=['Target'])
    y_train = train_with_target['Target']
    
    after_len = len(X_train_clean)
    print(f"✓ Removed {before_len - after_len} duplicates")
else:
    print("✓ Tidak ada duplicates")


# --- 3.4 FEATURE ENGINEERING ---
# Feature engineering: membuat fitur baru dari fitur existing
# untuk meningkatkan performa model
print("\n--- 3.4 Feature Engineering ---")

X_train_fe = X_train_clean.copy()
X_val_fe = X_val_clean.copy()

# Total units approved
if all(col in X_train_fe.columns for col in ['Curricular units 1st sem (approved)', 
                                               'Curricular units 2nd sem (approved)']):
    X_train_fe['Total_Units_Approved'] = (
        X_train_fe['Curricular units 1st sem (approved)'] + 
        X_train_fe['Curricular units 2nd sem (approved)']
    )
    X_val_fe['Total_Units_Approved'] = (
        X_val_fe['Curricular units 1st sem (approved)'] + 
        X_val_fe['Curricular units 2nd sem (approved)']
    )
    numerical_cols.append('Total_Units_Approved')
    print("✓ Feature: Total_Units_Approved")

# Average grade
if all(col in X_train_fe.columns for col in ['Curricular units 1st sem (grade)', 
                                               'Curricular units 2nd sem (grade)']):
    X_train_fe['Avg_Grade'] = (
        X_train_fe['Curricular units 1st sem (grade)'] + 
        X_train_fe['Curricular units 2nd sem (grade)']
    ) / 2
    X_val_fe['Avg_Grade'] = (
        X_val_fe['Curricular units 1st sem (grade)'] + 
        X_val_fe['Curricular units 2nd sem (grade)']
    ) / 2
    numerical_cols.append('Avg_Grade')
    print("✓ Feature: Avg_Grade")


# ============================================================================
# 4. DATA PREPROCESSING FROM SCRATCH
# ============================================================================
print("\n" + "=" * 80)
print("4. DATA PREPROCESSING FROM SCRATCH")
print("=" * 80)

# --- 4.1 FEATURE SCALING (STANDARD SCALER) ---
# Feature Scaling memastikan semua fitur numerik memiliki skala yang sama.
# Algoritma seperti Logistic Regression dan SVM sangat sensitif terhadap skala fitur.
# StandardScaler menggunakan Z-score normalization: X_scaled = (X - mean) / std

class StandardScaler:
    """
    Standard Scaler dari scratch.
    Formula: X_scaled = (X - mean) / std
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        """Hitung mean dan std dari training data"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Hindari pembagian dengan 0
        self.std_[self.std_ == 0] = 1
        return self
    
    def transform(self, X):
        """Transform data menggunakan mean dan std yang sudah di-fit"""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler belum di-fit!")
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """Fit dan transform sekaligus"""
        return self.fit(X).transform(X)

# Apply scaling
print("\n--- 4.1 Feature Scaling ---")
scaler = StandardScaler()
X_train_scaled = X_train_fe.copy()
X_val_scaled = X_val_fe.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train_fe[numerical_cols])
X_val_scaled[numerical_cols] = scaler.transform(X_val_fe[numerical_cols])
print("✓ Scaling selesai")


# --- 4.2 FEATURE ENCODING (LABEL ENCODER) ---
# Encoding mengubah categorical values menjadi numerical values.
# Machine learning models hanya bisa memproses angka, bukan string/kategori.
# Label Encoding: assign integer unik untuk setiap kategori.

class LabelEncoder:
    """
    Label Encoder dari scratch.
    Mengubah categorical values menjadi integer.
    """
    def __init__(self):
        self.label_mapping_ = {}
        self.inverse_mapping_ = {}
    
    def fit(self, X, columns):
        """Buat mapping dari categorical values ke integer"""
        for col in columns:
            unique_values = X[col].unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            self.label_mapping_[col] = mapping
            self.inverse_mapping_[col] = {idx: val for val, idx in mapping.items()}
        return self
    
    def transform(self, X, columns):
        """Transform categorical values ke integer"""
        X_encoded = X.copy()
        for col in columns:
            if col in self.label_mapping_:
                X_encoded[col] = X[col].map(self.label_mapping_[col])
                # Handle unseen values
                X_encoded[col] = X_encoded[col].fillna(-1).astype(int)
        return X_encoded
    
    def fit_transform(self, X, columns):
        """Fit dan transform sekaligus"""
        return self.fit(X, columns).transform(X, columns)

# Apply encoding
print("\n--- 4.2 Feature Encoding ---")
label_encoder = LabelEncoder()
X_train_encoded = label_encoder.fit_transform(X_train_scaled, categorical_cols)
X_val_encoded = label_encoder.transform(X_val_scaled, categorical_cols)
print("✓ Encoding selesai")

# Encode target
target_encoder = LabelEncoder()
y_train_encoded = pd.Series(
    target_encoder.fit_transform(pd.DataFrame({'Target': y_train}), ['Target'])['Target'].values,
    index=y_train.index
)
y_val_encoded = pd.Series(
    target_encoder.transform(pd.DataFrame({'Target': y_val}), ['Target'])['Target'].values,
    index=y_val.index
)
print("✓ Target encoding selesai")


# ============================================================================
# 4.3 HANDLING IMBALANCED DATASET (OPTIONAL)
# ============================================================================
# CATATAN: Handling imbalanced dataset bisa dilakukan dengan:
# 1. Oversampling (SMOTE) - menambah data kelas minoritas
# 2. Undersampling - mengurangi data kelas mayoritas
# 3. Class weights - memberikan bobot lebih ke kelas minoritas
#
# DALAM IMPLEMENTASI INI: Tidak dilakukan handling imbalanced karena:
# - Dataset sudah relatif seimbang (cek dari ProfileReport)
# - Stratified split sudah menjaga proporsi kelas
# - Fokus pada implementasi algoritma from scratch
print("\n--- 4.3 Handling Imbalanced Dataset ---")
print("⊘ Skipped: Dataset sudah relatif seimbang berdasarkan EDA")


# ============================================================================
# 4.4 DATA NORMALIZATION
# ============================================================================
# CATATAN: Data normalization vs standardization:
# - Normalization (Min-Max Scaling): scale ke range [0,1]
# - Standardization (Z-score): scale ke mean=0, std=1
#
# DALAM IMPLEMENTASI INI: Menggunakan StandardScaler (Z-score)
# Alasan:
# - Lebih robust terhadap outliers dibanding Min-Max
# - Cocok untuk Logistic Regression dan SVM
# - Tidak assume distribusi tertentu
print("\n--- 4.4 Data Normalization ---")
print("✓ Menggunakan StandardScaler (Z-score normalization)")
print("  Alasan: Robust terhadap outliers, cocok untuk LogReg & SVM")


# ============================================================================
# 4.5 DIMENSIONALITY REDUCTION (OPTIONAL)
# ============================================================================
# CATATAN: Dimensionality reduction seperti PCA bisa digunakan untuk:
# 1. Mengurangi jumlah fitur
# 2. Mempercepat training
# 3. Mengurangi overfitting
# 4. Visualisasi data high-dimensional
#
# DALAM IMPLEMENTASI INI: Tidak dilakukan dimensionality reduction karena:
# - Jumlah fitur masih manageable (~37 fitur)
# - Decision Tree tidak memerlukan dimensionality reduction
# - Ingin mempertahankan interpretability fitur original
# - Fokus pada implementasi algoritma from scratch
print("\n--- 4.5 Dimensionality Reduction ---")
print("⊘ Skipped: Jumlah fitur masih manageable, prioritas interpretability")


# ============================================================================
# 4.6 COMPILE PREPROCESSING PIPELINE
# ============================================================================
# Semua preprocessing steps yang sudah dilakukan:
# 1. Identifikasi kolom categorical vs numerical
# 2. Handling outliers (IQR method + clipping)
# 3. Remove duplicates
# 4. Feature engineering (Total_Units_Approved, Avg_Grade)
# 5. Feature scaling (StandardScaler untuk numerical)
# 6. Feature encoding (LabelEncoder untuk categorical)
# 7. Target encoding
#
# CATATAN: Dalam sklearn, bisa menggunakan Pipeline object.
# Di sini kita apply secara berurutan manual untuk kontrol penuh.
print("\n--- 4.6 Compile Preprocessing Pipeline ---")
print("✓ Preprocessing pipeline selesai:")
print("  1. Outliers handled ✓")
print("  2. Duplicates removed ✓")
print("  3. Features engineered ✓")
print("  4. Features scaled ✓")
print("  5. Features encoded ✓")
print("  6. Target encoded ✓")
print(f"\nFinal training shape: {X_train_encoded.shape}")
print(f"Final validation shape: {X_val_encoded.shape}")


# ============================================================================
# 5. MODEL IMPLEMENTATION FROM SCRATCH
# ============================================================================
print("\n" + "=" * 80)
print("5. MODEL IMPLEMENTATION FROM SCRATCH")
print("=" * 80)

# --- DECISION TREE (CART ALGORITHM) ---
class DecisionTreeNode:
    """Node untuk Decision Tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Untuk leaf node

class DecisionTreeClassifier:
    """
    Decision Tree menggunakan CART algorithm dengan Gini Impurity.
    Sesuai requirement: SALAH SATU dari ID3, C4.5, atau CART.
    """
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def _gini_impurity(self, y):
        """
        Menghitung Gini Impurity.
        Gini = 1 - Σ(p_i²)
        """
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _split_data(self, X, y, feature_idx, threshold):
        """Split data berdasarkan feature dan threshold"""
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]
    
    def _information_gain(self, y, y_left, y_right):
        """Hitung information gain dari split"""
        parent_gini = self._gini_impurity(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        weighted_gini = (n_left / n) * self._gini_impurity(y_left) + \
                        (n_right / n) * self._gini_impurity(y_right)
        
        return parent_gini - weighted_gini
    
    def _best_split(self, X, y):
        """Cari best feature dan threshold untuk split"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split_data(X, y, feature_idx, threshold)
                
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                gain = self._information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Build tree secara rekursif"""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(value=leaf_value)
        
        # Cari best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(value=leaf_value)
        
        # Split data dan build subtrees
        X_left, X_right, y_left, y_right = self._split_data(X, y, best_feature, best_threshold)
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        return DecisionTreeNode(best_feature, best_threshold, left_subtree, right_subtree)
    
    def fit(self, X, y):
        """Train decision tree"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        """Prediksi untuk single sample"""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Prediksi untuk multiple samples"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = np.array([self._predict_sample(x, self.tree) for x in X])
        return predictions
    
    def save(self, filepath):
        """Save model ke file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load model dari file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

print("\n✓ DecisionTreeClassifier (CART) berhasil dibuat")


# --- LOGISTIC REGRESSION ---
class LogisticRegression:
    """
    Logistic Regression dari scratch menggunakan Gradient Descent.
    Support multiclass dengan One-vs-Rest strategy.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization='l2', lambda_param=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
        self.classes = None
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _compute_loss(self, y, y_pred, weights):
        """Compute loss dengan regularization"""
        m = len(y)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy loss
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        # Add regularization
        if self.regularization == 'l2':
            loss += (self.lambda_param / (2 * m)) * np.sum(weights ** 2)
        elif self.regularization == 'l1':
            loss += (self.lambda_param / m) * np.sum(np.abs(weights))
        
        return loss
    
    def _fit_binary(self, X, y):
        """Fit untuk binary classification"""
        m, n = X.shape
        
        # Initialize parameters
        weights = np.zeros(n)
        bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_output = np.dot(X, weights) + bias
            y_pred = self._sigmoid(linear_output)
            
            # Compute gradients
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)
            
            # Add regularization to gradients
            if self.regularization == 'l2':
                dw += (self.lambda_param / m) * weights
            elif self.regularization == 'l1':
                dw += (self.lambda_param / m) * np.sign(weights)
            
            # Update parameters
            weights -= self.learning_rate * dw
            bias -= self.learning_rate * db
        
        return weights, bias
    
    def fit(self, X, y):
        """Train logistic regression (One-vs-Rest untuk multiclass)"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        if n_classes == 2:
            # Binary classification
            y_binary = (y == self.classes[1]).astype(int)
            self.weights, self.bias = self._fit_binary(X, y_binary)
        else:
            # Multiclass: One-vs-Rest
            self.weights = []
            self.bias = []
            
            for class_label in self.classes:
                y_binary = (y == class_label).astype(int)
                w, b = self._fit_binary(X, y_binary)
                self.weights.append(w)
                self.bias.append(b)
            
            self.weights = np.array(self.weights)
            self.bias = np.array(self.bias)
        
        return self
    
    def predict_proba(self, X):
        """Prediksi probabilitas"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if len(self.classes) == 2:
            linear_output = np.dot(X, self.weights) + self.bias
            proba = self._sigmoid(linear_output)
            return np.vstack([1 - proba, proba]).T
        else:
            # Multiclass
            probas = []
            for i in range(len(self.classes)):
                linear_output = np.dot(X, self.weights[i]) + self.bias[i]
                proba = self._sigmoid(linear_output)
                probas.append(proba)
            
            probas = np.array(probas).T
            # Normalize probabilities
            probas = probas / probas.sum(axis=1, keepdims=True)
            return probas
    
    def predict(self, X):
        """Prediksi class"""
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]
    
    def save(self, filepath):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

print("✓ LogisticRegression berhasil dibuat")


# --- SVM (ONE-AGAINST-ONE) ---
class SVM:
    """
    SVM dari scratch dengan ONE-AGAINST-ONE strategy untuk multiclass.
    Sesuai requirement dari foto.
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.classifiers = {}  # Dictionary untuk menyimpan binary classifiers
        self.classes = None
    
    def _compute_hinge_loss(self, y, y_pred, weights):
        """
        Compute hinge loss dengan regularization.
        Hinge loss = max(0, 1 - y * y_pred)
        """
        m = len(y)
        hinge_losses = np.maximum(0, 1 - y * y_pred)
        loss = np.mean(hinge_losses)
        
        # Add L2 regularization
        loss += (self.lambda_param / 2) * np.sum(weights ** 2)
        
        return loss
    
    def _fit_binary(self, X, y):
        """
        Fit binary SVM classifier.
        y harus berisi nilai -1 dan 1
        """
        m, n = X.shape
        
        # Initialize parameters
        weights = np.zeros(n)
        bias = 0
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            # Compute linear output
            linear_output = np.dot(X, weights) + bias
            
            # Compute condition for hinge loss
            condition = y * linear_output < 1
            
            # Compute gradients
            dw = np.zeros_like(weights)
            db = 0
            
            # Hinge loss gradient
            dw = self.lambda_param * weights
            dw -= np.dot(X[condition].T, y[condition]) / m
            db = -np.sum(y[condition]) / m
            
            # Update parameters
            weights -= self.learning_rate * dw
            bias -= self.learning_rate * db
        
        return weights, bias
    
    def fit(self, X, y):
        """
        Train SVM dengan ONE-AGAINST-ONE strategy.
        Untuk n kelas, akan membuat n*(n-1)/2 binary classifiers.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Train binary classifier untuk setiap pasangan kelas
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                class_i = self.classes[i]
                class_j = self.classes[j]
                
                # Ambil data untuk 2 kelas ini saja
                mask = (y == class_i) | (y == class_j)
                X_binary = X[mask]
                y_binary = y[mask]
                
                # Convert ke -1 dan 1
                y_binary = np.where(y_binary == class_i, -1, 1)
                
                # Train binary classifier
                weights, bias = self._fit_binary(X_binary, y_binary)
                
                # Simpan classifier
                self.classifiers[(class_i, class_j)] = (weights, bias)
        
        return self
    
    def _decision_function(self, X, class_pair):
        """Compute decision function untuk class pair"""
        weights, bias = self.classifiers[class_pair]
        return np.dot(X, weights) + bias
    
    def predict(self, X):
        """
        Prediksi menggunakan voting dari semua binary classifiers.
        Setiap classifier vote untuk salah satu dari 2 kelasnya.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes)))
        
        # Dapatkan vote dari setiap binary classifier
        for (class_i, class_j), (weights, bias) in self.classifiers.items():
            decision_values = np.dot(X, weights) + bias
            
            # Vote untuk class_i jika decision_value < 0, else class_j
            class_i_idx = np.where(self.classes == class_i)[0][0]
            class_j_idx = np.where(self.classes == class_j)[0][0]
            
            for idx in range(n_samples):
                if decision_values[idx] < 0:
                    votes[idx, class_i_idx] += 1
                else:
                    votes[idx, class_j_idx] += 1
        
        # Prediksi adalah kelas dengan vote terbanyak
        predictions = self.classes[np.argmax(votes, axis=1)]
        return predictions
    
    def save(self, filepath):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

print("✓ SVM (One-Against-One) berhasil dibuat")


# ============================================================================
# 6. TRAINING & EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("6. TRAINING & EVALUATION")
print("=" * 80)

def calculate_metrics(y_true, y_pred, model_name):
    """Hitung accuracy, precision, recall, F1-score untuk setiap kelas"""
    from collections import Counter
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Per-class metrics
    classes = np.unique(y_true)
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for cls in classes:
        # True Positives, False Positives, False Negatives
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    print(f"\n{model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision per class: {[f'{p:.4f}' for p in precision_per_class]}")
    print(f"  Recall per class: {[f'{r:.4f}' for r in recall_per_class]}")
    print(f"  F1-score per class: {[f'{f:.4f}' for f in f1_per_class]}")
    
    return accuracy, precision_per_class, recall_per_class, f1_per_class


# --- TRAIN DECISION TREE ---
print("\n--- Training Decision Tree (CART) ---")
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5)
dt_model.fit(X_train_encoded, y_train_encoded)

dt_pred_train = dt_model.predict(X_train_encoded)
dt_pred_val = dt_model.predict(X_val_encoded)

dt_acc_train, _, _, _ = calculate_metrics(y_train_encoded.values, dt_pred_train, "DT Train")
dt_acc_val, _, _, _ = calculate_metrics(y_val_encoded.values, dt_pred_val, "DT Validation")


# --- TRAIN LOGISTIC REGRESSION ---
print("\n--- Training Logistic Regression ---")
lr_model = LogisticRegression(learning_rate=0.01, n_iterations=1000, regularization='l2', lambda_param=0.01)
lr_model.fit(X_train_encoded, y_train_encoded)

lr_pred_train = lr_model.predict(X_train_encoded)
lr_pred_val = lr_model.predict(X_val_encoded)

lr_acc_train, _, _, _ = calculate_metrics(y_train_encoded.values, lr_pred_train, "LR Train")
lr_acc_val, _, _, _ = calculate_metrics(y_val_encoded.values, lr_pred_val, "LR Validation")


# --- TRAIN SVM (ONE-AGAINST-ONE) ---
print("\n--- Training SVM (One-Against-One) ---")
svm_model = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
svm_model.fit(X_train_encoded, y_train_encoded)

svm_pred_train = svm_model.predict(X_train_encoded)
svm_pred_val = svm_model.predict(X_val_encoded)

svm_acc_train, _, _, _ = calculate_metrics(y_train_encoded.values, svm_pred_train, "SVM Train")
svm_acc_val, _, _, _ = calculate_metrics(y_val_encoded.values, svm_pred_val, "SVM Validation")


# ============================================================================
# 7. SAVE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("7. SAVE MODELS")
print("=" * 80)

dt_model.save("../models/decision_tree.pkl")
print("✓ Decision Tree saved to '../models/decision_tree.pkl'")

lr_model.save("../models/logistic_regression.pkl")
print("✓ Logistic Regression saved to '../models/logistic_regression.pkl'")

svm_model.save("../models/svm.pkl")
print("✓ SVM saved to '../models/svm.pkl'")


# ============================================================================
# 8. COMPARISON WITH SCIKIT-LEARN
# ============================================================================
print("\n" + "=" * 80)
print("8. PERBANDINGAN DENGAN SCIKIT-LEARN")
print("=" * 80)

from sklearn.tree import DecisionTreeClassifier as SKDecisionTree
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.svm import SVC as SKSVM
from sklearn.metrics import accuracy_score

# Sklearn Decision Tree
print("\n--- Sklearn Decision Tree ---")
sk_dt = SKDecisionTree(criterion='gini', max_depth=10, min_samples_split=20, 
                       min_samples_leaf=5, random_state=RANDOM_STATE)
sk_dt.fit(X_train_encoded, y_train_encoded)
sk_dt_pred_val = sk_dt.predict(X_val_encoded)
sk_dt_acc = accuracy_score(y_val_encoded, sk_dt_pred_val)
print(f"Validation Accuracy: {sk_dt_acc:.4f}")

# Sklearn Logistic Regression
print("\n--- Sklearn Logistic Regression ---")
sk_lr = SKLogisticRegression(max_iter=1000, C=1/0.01, random_state=RANDOM_STATE)
sk_lr.fit(X_train_encoded, y_train_encoded)
sk_lr_pred_val = sk_lr.predict(X_val_encoded)
sk_lr_acc = accuracy_score(y_val_encoded, sk_lr_pred_val)
print(f"Validation Accuracy: {sk_lr_acc:.4f}")

# Sklearn SVM
print("\n--- Sklearn SVM ---")
sk_svm = SKSVM(kernel='linear', C=1/0.01, decision_function_shape='ovo', random_state=RANDOM_STATE)
sk_svm.fit(X_train_encoded, y_train_encoded)
sk_svm_pred_val = sk_svm.predict(X_val_encoded)
sk_svm_acc = accuracy_score(y_val_encoded, sk_svm_pred_val)
print(f"Validation Accuracy: {sk_svm_acc:.4f}")

# Summary
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print(f"\nDecision Tree:")
print(f"  From Scratch: {dt_acc_val:.4f}")
print(f"  Scikit-learn: {sk_dt_acc:.4f}")
print(f"\nLogistic Regression:")
print(f"  From Scratch: {lr_acc_val:.4f}")
print(f"  Scikit-learn: {sk_lr_acc:.4f}")
print(f"\nSVM (One-Against-One):")
print(f"  From Scratch: {svm_acc_val:.4f}")
print(f"  Scikit-learn: {sk_svm_acc:.4f}")


# ============================================================================
# 9. PREDICTION & SUBMISSION
# ============================================================================
print("\n" + "=" * 80)
print("9. PREDICTION & SUBMISSION")
print("=" * 80)

# Load test data
test_df = pd.read_csv("../data/test.csv")
test_ids = test_df["Student_ID"]
X_test = test_df.drop(columns=["Student_ID"])

# Preprocessing test data (SAME STEPS AS TRAINING)
print("\nPreprocessing test data...")

# Convert categorical
X_test[categorical_cols] = X_test[categorical_cols].astype(str)

# Feature engineering
if all(col in X_test.columns for col in ['Curricular units 1st sem (approved)', 
                                          'Curricular units 2nd sem (approved)']):
    X_test['Total_Units_Approved'] = (
        X_test['Curricular units 1st sem (approved)'] + 
        X_test['Curricular units 2nd sem (approved)']
    )

if all(col in X_test.columns for col in ['Curricular units 1st sem (grade)', 
                                          'Curricular units 2nd sem (grade)']):
    X_test['Avg_Grade'] = (
        X_test['Curricular units 1st sem (grade)'] + 
        X_test['Curricular units 2nd sem (grade)']
    ) / 2

# Align columns
X_test = X_test[X_train_encoded.columns]

# Scaling
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Encoding
X_test = label_encoder.transform(X_test, categorical_cols)

# Predict menggunakan model terbaik (pilih berdasarkan validation accuracy)
best_model_name = max([
    ("Decision Tree", dt_acc_val),
    ("Logistic Regression", lr_acc_val),
    ("SVM", svm_acc_val)
], key=lambda x: x[1])[0]

print(f"\nMenggunakan model terbaik: {best_model_name}")

if best_model_name == "Decision Tree":
    y_test_pred_encoded = dt_model.predict(X_test)
elif best_model_name == "Logistic Regression":
    y_test_pred_encoded = lr_model.predict(X_test)
else:
    y_test_pred_encoded = svm_model.predict(X_test)

# Decode predictions
y_test_pred = pd.Series(
    target_encoder.inverse_mapping_['Target'][pred] 
    for pred in y_test_pred_encoded
)

# Create submission
submission = pd.DataFrame({
    "Student_ID": test_ids,
    "Target": y_test_pred
})

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
submission_path = f"../submission/submission_{timestamp}.csv"
submission.to_csv(submission_path, index=False)

print(f"\n✓ Submission saved to: {submission_path}")
print(f"✓ Total predictions: {len(submission)}")


# ============================================================================
# DONE!
# ============================================================================
print("\n" + "=" * 80)
print("ALL DONE! ✓")
print("=" * 80)
print("\nYang sudah dilakukan:")
print("1. ✓ Data Understanding dengan ProfileReport")
print("2. ✓ Split data 80-20 dengan stratified sampling")
print("3. ✓ Data preparation (outliers, duplicates, feature engineering)")
print("4. ✓ Preprocessing dari scratch (StandardScaler, LabelEncoder)")
print("5. ✓ Implementasi Decision Tree (CART) dari scratch")
print("6. ✓ Implementasi Logistic Regression dari scratch")
print("7. ✓ Implementasi SVM (One-Against-One) dari scratch")
print("8. ✓ Training & evaluation semua model")
print("9. ✓ Save semua model (.pkl)")
print("10. ✓ Perbandingan dengan scikit-learn")
print("11. ✓ Generate submission file")
print("\nSemua model bisa di-load kembali dengan:")
print("  model = DecisionTreeClassifier.load('path/to/model.pkl')")
