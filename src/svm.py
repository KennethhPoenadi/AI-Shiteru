"""
SVM Implementation with Multiple Multiclass Strategies
- Binary SVM using gradient descent with hinge loss
- One-vs-Rest (OvR) multiclass strategy
- One-vs-One (OvO) multiclass strategy with voting
- DAGSVM with tournament-style elimination
"""

import numpy as np
import pandas as pd
from itertools import combinations

# ========================
# SVM BASE CLASSIFIER
# ========================

class BinarySVM:
    """Binary SVM classifier using gradient descent with hinge loss"""
    
    def __init__(self, lr=0.01, n_iters=5000, C=1.0, lambda_param=0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.C = C
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                
                if condition:
                    # Correctly classified: only apply regularization
                    self.weights -= self.lr * (self.lambda_param * self.weights)
                else:
                    # Misclassified: apply regularization + hinge loss gradient
                    self.weights -= self.lr * (self.lambda_param * self.weights - self.C * y_[idx] * x_i)
                    self.bias -= self.lr * (self.C * y_[idx])
    
    def decision_function(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        return np.sign(self.decision_function(X))


# ========================
# SVM: ONE-VS-REST
# ========================

class SVM_OneVsRest:
    """One-vs-Rest SVM: K binary classifiers"""
    
    def __init__(self, lr=0.01, n_iters=5000, C=1.0, lambda_param=0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.C = C
        self.lambda_param = lambda_param
        self.classifiers = {}
        self.classes = None
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        
        print(f"Training SVM One-vs-Rest with {len(self.classes)} classifiers...")
        
        for cls in self.classes:
            y_binary = np.where(y_arr == cls, 1, -1)
            clf = BinarySVM(self.lr, self.n_iters, self.C, self.lambda_param)
            clf.fit(X_arr, y_binary)
            self.classifiers[cls] = clf
            
    def predict(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        decision_values = np.zeros((X_arr.shape[0], len(self.classes)))
        
        for idx, cls in enumerate(self.classes):
            decision_values[:, idx] = self.classifiers[cls].decision_function(X_arr)
        
        return self.classes[np.argmax(decision_values, axis=1)]


# ========================
# SVM: ONE-VS-ONE
# ========================

class SVM_OneVsOne:
    """One-vs-One SVM: K(K-1)/2 binary classifiers with voting"""
    
    def __init__(self, lr=0.01, n_iters=5000, C=1.0, lambda_param=0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.C = C
        self.lambda_param = lambda_param
        self.classifiers = {}
        self.classes = None
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        
        n_classifiers = len(list(combinations(self.classes, 2)))
        print(f"Training SVM One-vs-One with {n_classifiers} classifiers...")
        
        for cls1, cls2 in combinations(self.classes, 2):
            mask = (y_arr == cls1) | (y_arr == cls2)
            X_pair = X_arr[mask]
            y_pair = np.where(y_arr[mask] == cls1, 1, -1)
            
            clf = BinarySVM(self.lr, self.n_iters, self.C, self.lambda_param)
            clf.fit(X_pair, y_pair)
            self.classifiers[(cls1, cls2)] = clf
            
    def predict(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        n_samples = X_arr.shape[0]
        votes = np.zeros((n_samples, len(self.classes)))
        
        for (cls1, cls2), clf in self.classifiers.items():
            predictions = clf.predict(X_arr)
            cls1_idx = np.where(self.classes == cls1)[0][0]
            cls2_idx = np.where(self.classes == cls2)[0][0]
            
            votes[predictions == 1, cls1_idx] += 1
            votes[predictions == -1, cls2_idx] += 1
        
        return self.classes[np.argmax(votes, axis=1)]


# ========================
# SVM: DAGSVM
# ========================

class SVM_DAGSVM:
    """DAGSVM: Tournament-style elimination structure"""
    
    def __init__(self, lr=0.01, n_iters=5000, C=1.0, lambda_param=0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.C = C
        self.lambda_param = lambda_param
        self.classifiers = {}
        self.classes = None
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        
        n_classifiers = len(list(combinations(self.classes, 2)))
        print(f"Training SVM DAGSVM with {n_classifiers} classifiers...")
        
        for cls1, cls2 in combinations(self.classes, 2):
            mask = (y_arr == cls1) | (y_arr == cls2)
            X_pair = X_arr[mask]
            y_pair = np.where(y_arr[mask] == cls1, 1, -1)
            
            clf = BinarySVM(self.lr, self.n_iters, self.C, self.lambda_param)
            clf.fit(X_pair, y_pair)
            self.classifiers[(cls1, cls2)] = clf
            
    def predict(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        n_samples = X_arr.shape[0]
        predictions = []
        
        for i in range(n_samples):
            x = X_arr[i:i+1]
            remaining = list(self.classes)
            
            while len(remaining) > 1:
                cls1, cls2 = remaining[0], remaining[1]
                
                if (cls1, cls2) in self.classifiers:
                    clf = self.classifiers[(cls1, cls2)]
                    pred = clf.predict(x)[0]
                    if pred == 1:
                        remaining.remove(cls2)
                    else:
                        remaining.remove(cls1)
                else:
                    clf = self.classifiers[(cls2, cls1)]
                    pred = clf.predict(x)[0]
                    if pred == 1:
                        remaining.remove(cls1)
                    else:
                        remaining.remove(cls2)
                        
            predictions.append(remaining[0])
            
        return np.array(predictions)

