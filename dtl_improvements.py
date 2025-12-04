"""
DTL Improvements untuk notebookDTL.ipynb
Copy-paste code di bawah ke notebook cells yang sesuai
Setiap section dipisahkan dengan === untuk menandai cell boundary
"""

# ====================================================================================================
# CELL 1: UPDATE DecisionTreeClassifier - ADD METHOD _get_used_features
# ====================================================================================================
# Tambahkan method ini di dalam class DecisionTreeClassifier, setelah method _count_leaves()

def _get_used_features(self, node, features_set=None):
    """
    Get set of all features yang digunakan di tree.
    """
    if features_set is None:
        features_set = set()
    
    if node is None or node.value is not None:
        return features_set
    
    # Add current node's feature
    if node.feature is not None:
        features_set.add(node.feature)
    
    # Recurse to children
    self._get_used_features(node.left, features_set)
    self._get_used_features(node.right, features_set)
    
    return features_set


# ====================================================================================================
# CELL 2: UPDATE DecisionTreeClassifier - UPDATE get_tree_info METHOD
# ====================================================================================================
# Replace method get_tree_info() yang sudah ada dengan yang ini:

def get_tree_info(self):
    """
    Get informasi tentang tree structure.
    """
    used_features = self._get_used_features(self.tree)
    
    return {
        'depth': self._get_tree_depth(self.tree),
        'n_leaves': self._count_leaves(self.tree),
        'n_features': self.n_features,
        'n_features_used': len(used_features),
        'used_features': sorted(list(used_features)),
        'n_classes': self.n_classes,
        'criterion': self.criterion,
        'ccp_alpha': self.ccp_alpha,
        'class_weighted': self.class_weights_ is not None
    }


# ====================================================================================================
# CELL 3: ADD DecisionTreeClassifier - TREE VISUALIZATION METHODS
# ====================================================================================================
# Tambahkan methods ini di dalam class DecisionTreeClassifier, sebelum save()

def print_tree(self, node=None, feature_names=None, depth=0, prefix="", is_left=True):
    """
    Print tree structure in text format.
    
    Parameters:
    -----------
    node : DecisionTreeNode
        Current node to print
    feature_names : list
        List of feature names
    depth : int
        Current depth
    prefix : str
        Prefix for indentation
    is_left : bool
        Whether this is left child
    """
    if node is None:
        node = self.tree
    
    if node.value is not None:
        # Leaf node
        print(f"{prefix}{'L' if is_left else 'R'}--- LEAF: Class={node.value} (samples={node.samples})")
        return
    
    # Internal node
    feature_name = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
    print(f"{prefix}{'L' if is_left else 'R'}--- {feature_name} <= {node.threshold:.2f}? (samples={node.samples})")
    
    # Print left subtree (True branch)
    if node.left:
        extension = "|   " if node.right else "    "
        self.print_tree(node.left, feature_names, depth + 1, prefix + extension, True)
    
    # Print right subtree (False branch)
    if node.right:
        extension = "    "
        self.print_tree(node.right, feature_names, depth + 1, prefix + extension, False)

def visualize_tree_summary(self, feature_names=None, max_depth_show=3):
    """
    Print concise tree summary showing structure up to max_depth_show.
    """
    print("\n" + "="*100)
    print("TREE STRUCTURE VISUALIZATION")
    print("="*100)
    
    tree_info = self.get_tree_info()
    print(f"Tree Statistics:")
    print(f"  - Actual Depth: {tree_info['depth']}")
    print(f"  - Number of Leaves: {tree_info['n_leaves']}")
    print(f"  - Features Used: {tree_info['n_features_used']}/{tree_info['n_features']}")
    
    print(f"\nTree Structure (showing depth 0-{max_depth_show}):")
    print("-"*100)
    self._print_tree_limited(self.tree, feature_names, 0, max_depth_show, "", True)
    
    if tree_info['depth'] > max_depth_show:
        print(f"\n... (tree continues to depth {tree_info['depth']})")
    print("="*100)

def _print_tree_limited(self, node, feature_names, depth, max_depth, prefix, is_left):
    """Helper function to print tree up to max_depth."""
    if node is None or depth > max_depth:
        if depth > max_depth and node.value is None:
            print(f"{prefix}{'L' if is_left else 'R'}--- ...")
        return
    
    if node.value is not None:
        print(f"{prefix}{'L' if is_left else 'R'}--- LEAF: Class={node.value} (n={node.samples})")
        return
    
    feature_name = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
    print(f"{prefix}{'L' if is_left else 'R'}--- {feature_name} <= {node.threshold:.2f}?")
    
    if node.left:
        extension = "|   " if node.right else "    "
        self._print_tree_limited(node.left, feature_names, depth + 1, max_depth, prefix + extension, True)
    
    if node.right:
        extension = "    "
        self._print_tree_limited(node.right, feature_names, depth + 1, max_depth, prefix + extension, False)


# ====================================================================================================
# CELL 4: IMPROVED DEPTH EXPLORATION - REPLACE EXISTING DEPTH EXPLORATION CELL
# ====================================================================================================

print("\n" + "="*100)
print("QUICK DEPTH EXPLORATION - Find optimal max_depth")
print("="*100)

print("Strategy: Test different max_depth values with SMOTE balancing")
print("Fixed params: min_samples_split=20, min_samples_leaf=10, criterion=entropy")
print("\nNote:")
print("  - Using SMOTE to balance classes before each training")
print("  - Testing depths: [5, 8, 10, 12, 15, 18, 20, 25]")
print("  - Total features: 44 (original + engineered)")
print("  - Depth = max questions in decision path")
print("  - Features used = how many attributes tree actually needs")
print("-"*100)

# Get feature names
feature_names = list(X_train_encoded.columns)

depth_results = []
test_depths = [5, 8, 10, 12, 15, 18, 20, 25]

for depth in test_depths:
    # Apply SMOTE
    smote = SMOTE(k_neighbors=5, sampling_strategy='auto', random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_final, y_train_encoded)
    
    # Train model
    dt = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_split=20,
        min_samples_leaf=10,
        criterion='entropy'
    )
    dt.fit(X_train_resampled, y_train_resampled)
    
    # Predictions
    pred_val = dt.predict(X_val_final)
    pred_train = dt.predict(X_train_final)
    
    # Metrics
    acc_val = np.mean(y_val_encoded.values == pred_val)
    acc_train = np.mean(y_train_encoded.values == pred_train)
    
    info = dt.get_tree_info()
    overfitting = acc_train - acc_val
    
    # Print results
    print(f"Depth={depth:2d}: Train={acc_train:.4f}, Val={acc_val:.4f}, "
          f"Overfit={overfitting:+.4f}, Leaves={info['n_leaves']:3d}, "
          f"Features={info['n_features_used']:2d}/{info['n_features']:2d}")
    
    # Show preview of features used
    used_feature_names = [feature_names[idx] for idx in info['used_features']]
    print(f"          Features: {', '.join(used_feature_names[:5])}", end='')
    if len(used_feature_names) > 5:
        print(f", ... (+{len(used_feature_names)-5} more)")
    else:
        print()
    
    # Store results
    depth_results.append({
        'depth': depth,
        'acc_val': acc_val,
        'acc_train': acc_train,
        'overfitting': overfitting,
        'leaves': info['n_leaves'],
        'n_features_used': info['n_features_used'],
        'used_features': info['used_features'],
        'model': dt
    })

# Find best depth based on validation accuracy
best_depth_result = max(depth_results, key=lambda x: x['acc_val'])
best_depth = best_depth_result['depth']

print("\n" + "-"*100)
print(f"[OK] BEST DEPTH FOUND: {best_depth}")
print(f"  Validation Accuracy: {best_depth_result['acc_val']:.4f}")
print(f"  Training Accuracy: {best_depth_result['acc_train']:.4f}")
print(f"  Overfitting Gap: {best_depth_result['overfitting']:+.4f}")
print(f"  Number of Leaves: {best_depth_result['leaves']}")
print(f"  Features Used: {best_depth_result['n_features_used']}/{len(feature_names)}")

print(f"\n[Stats] Features used in best tree (sorted by importance):")
# Get feature importances
best_model = best_depth_result['model']
used_feature_indices = best_depth_result['used_features']
feature_importances = []
for idx in used_feature_indices:
    imp = best_model.feature_importances_[idx]
    name = feature_names[idx]
    feature_importances.append((name, imp))

# Sort by importance
feature_importances.sort(key=lambda x: x[1], reverse=True)

# Print top features
print(f"  Top 15 most important features:")
for i, (name, imp) in enumerate(feature_importances[:15], 1):
    print(f"    {i:2d}. {name:45s} = {imp:.4f}")

if len(feature_importances) > 15:
    print(f"  ... and {len(feature_importances) - 15} more features")

print("="*100)


# ====================================================================================================
# CELL 5: DETAILED FEATURES BREAKDOWN - ADD AFTER DEPTH EXPLORATION
# ====================================================================================================

print("\n" + "="*100)
print("DETAILED FEATURES BREAKDOWN BY DEPTH")
print("="*100)
print("Understanding the numbers:")
print("  - 'Depth=5 | Features Used: 19/44' means:")
print("    * Tree can ask max 5 questions in sequence (depth)")
print("    * But only needs 19 different attributes to make all decisions")
print("    * Other 25 attributes are not important for this tree structure")
print("  - Higher depth -> can use more features -> more complex model")
print("  - SMOTE was applied before training to balance classes")
print("="*100)

for result in depth_results:
    depth = result['depth']
    model = result['model']
    used_indices = result['used_features']
    
    # Get features with importance
    features_with_imp = []
    for idx in used_indices:
        imp = model.feature_importances_[idx]
        name = feature_names[idx]
        features_with_imp.append((name, imp))
    
    # Sort by importance
    features_with_imp.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'-'*100}")
    print(f"Depth={depth:2d} | Val Acc={result['acc_val']:.4f} | Features Used: {len(used_indices)}/{len(feature_names)} "
          f"| Unused: {len(feature_names) - len(used_indices)}")
    print(f"{'-'*100}")
    
    # Print top 10 features for this depth
    print(f"  Top 10 most important features (sorted by importance):")
    for i, (name, imp) in enumerate(features_with_imp[:10], 1):
        print(f"    {i:2d}. {name:45s} = {imp:.4f}")
    
    if len(features_with_imp) > 10:
        print(f"  ... and {len(features_with_imp) - 10} more features")

print("\n" + "="*100)

print(f"\n-> Best depth found: {best_depth} with Val Acc: {best_depth_result['acc_val']:.4f}")
print(f"  Overfitting: {best_depth_result['overfitting']:+.4f}")
print(f"  Leaves: {best_depth_result['leaves']}")

# Use best model from depth exploration
best_model = best_depth_result['model']
best_params = {
    'max_depth': best_depth,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'criterion': 'entropy'
}

print("\n" + "="*100)
print(f"SELECTED MODEL: max_depth={best_depth}")
print("="*100)


# ====================================================================================================
# CELL 6: VISUALIZE FINAL TREE STRUCTURE - ADD AFTER FINAL MODEL SECTION
# ====================================================================================================

# Visualize tree structure
print("\n" + "="*100)
print("TREE STRUCTURE VISUALIZATION")
print("="*100)

dt_model.visualize_tree_summary(feature_names=feature_names, max_depth_show=4)

print("\nInterpretation:")
print("  - L--- : Left branch (condition is TRUE, value <= threshold)")
print("  - R--- : Right branch (condition is FALSE, value > threshold)")
print("  - LEAF: Terminal node with final class prediction")
print("  - (n=X): Number of samples reaching that node")
print("\nKey insights:")
print("  - First split is most important (root node)")
print("  - Leaf nodes show final predictions")
print("  - Path from root to leaf = decision rules for that class")


# ====================================================================================================
# CELL 7: REMOVE UNICODE CHARACTERS SCRIPT - RUN ONCE
# ====================================================================================================

"""
Script to remove all Unicode/emoji characters from notebook.
Save this as remove_unicode_dtl.py and run once:
python remove_unicode_dtl.py
"""

# import re
# 
# replacements = {
#     'checkmark': '[OK]',
#     'chart': '[Stats]',
#     'arrow': '->',
#     'cross': '[SKIP]',
#     'note': '[Note]',
#     'bullet': '*',
#     'multiply': 'x',
# }
# 
# filepath = 'src/notebookDTL.ipynb'
# with open(filepath, 'r', encoding='utf-8') as f:
#     content = f.read()
# 
# # Replace Unicode characters
# content = content.replace('✓', '[OK]')
# content = content.replace('✅', '[OK]')
# content = content.replace('❌', '[X]')
# content = content.replace('⚠', '[!]')
# content = content.replace('📊', '[Stats]')
# content = content.replace('→', '->')
# content = content.replace('⊘', '[SKIP]')
# content = content.replace('📌', '[Note]')
# content = content.replace('•', '*')
# content = content.replace('×', 'x')
# 
# with open(filepath, 'w', encoding='utf-8') as f:
#     f.write(content)
# 
# print("Done! All Unicode characters replaced.")


# ====================================================================================================
# SUMMARY PERBEDAAN notebookDTL vs notebookDTLAlt
# ====================================================================================================

"""
=== PERBEDAAN FILE ===

notebookDTL.ipynb (FILE INI):
-------------------------------
1. [OK] Student_ID dropped
2. [OK] SMOTE untuk balancing classes
3. [OK] Depth exploration dengan SMOTE
4. [OK] Hyperparameter tuning penuh (grid search)
5. [OK] Cost-complexity pruning
6. [OK] Detailed feature tracking per depth
7. [OK] Tree visualization

Karakteristik:
- Lebih lengkap dengan SMOTE
- Grid search hyperparameter
- Cocok untuk imbalanced dataset
- Eksekusi lebih lama (karena SMOTE + grid search)

notebookDTLAlt.ipynb:
---------------------
1. [OK] Student_ID dropped
2. [X] NO SMOTE (faster execution)
3. [OK] Quick depth exploration (9 depths)
4. [OK] Fine-tuning hanya di akhir
5. [X] NO pruning
6. [OK] Lenient outlier handling
7. [OK] Tree visualization

Karakteristik:
- Lebih cepat eksekusi
- No SMOTE (assume classes balanced enough)
- Quick testing workflow
- Cocok untuk quick experiments

KAPAN PAKAI YANG MANA?
-----------------------
Gunakan notebookDTL jika:
- Dataset sangat imbalanced
- Punya waktu untuk full grid search
- Perlu comprehensive tuning

Gunakan notebookDTLAlt jika:
- Need quick results
- Classes relatively balanced
- Iterative experimentation
"""


# ====================================================================================================
# INSTRUCTIONS
# ====================================================================================================

"""
CARA PENGGUNAAN:
================

1. Buka notebookDTL.ipynb

2. CELL 1-3: Update DecisionTreeClassifier class
   - Tambahkan method _get_used_features() setelah _count_leaves()
   - Replace method get_tree_info() yang lama
   - Tambahkan methods print_tree(), visualize_tree_summary(), _print_tree_limited()

3. CELL 4: Replace cell "QUICK DEPTH EXPLORATION" yang sudah ada
   - Hapus cell lama yang punya depth exploration
   - Copy-paste code dari CELL 4 di atas

4. CELL 5: Tambahkan cell baru setelah depth exploration
   - Buat cell baru
   - Copy-paste code dari CELL 5 (DETAILED FEATURES BREAKDOWN)

5. CELL 6: Tambahkan cell baru setelah FINAL MODEL section
   - Cari section yang print final model info
   - Tambahkan cell baru di bawahnya
   - Copy-paste code dari CELL 6 (TREE VISUALIZATION)

6. CELL 7: (Optional) Remove Unicode
   - Uncomment code di CELL 7
   - Save as remove_unicode_dtl.py
   - Run: python remove_unicode_dtl.py

HASIL AKHIR:
============
- Depth exploration dengan feature tracking detail
- Breakdown features per depth level
- Tree structure visualization
- Clean ASCII output (no emoji)
- Comparison dengan notebookDTLAlt jelas

TESTING:
========
Setelah update, run notebook dari awal untuk verify:
1. Depth exploration menampilkan features used
2. Detailed breakdown muncul dengan benar
3. Tree visualization bekerja
4. No error di semua cell
"""
