"""
Error analysis utilities for classification models.

Provides:
- evaluate_model: prints classification report and returns metrics
- plot_confusion_matrix: matplotlib heatmap
- plot_calibration_curve: calibration curve and reliability diagram
- permutation_importance_features: permutation importance with sklearn
- shap_explain: SHAP explanations if shap package installed (fallback informative message)
- partial_dependence: simple partial dependence approximation (feature grid average prediction)

Usage:
>>> from error_analysis import evaluate_model, plot_confusion_matrix, shap_explain
>>> evaluate_model(y_true, y_pred, labels=class_names)
>>> plot_confusion_matrix(y_true, y_pred, labels=class_names, save_path='cm.png')

The functions accept scikit-learn-like estimators (must implement predict and predict_proba or decision_function where required).
"""

from typing import Optional, Sequence, Tuple, Dict, Union
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance


def evaluate_model(y_true, y_pred, labels: Optional[Sequence[str]] = None) -> Dict:
    """Print classification report and return summary metrics.

    Returns a dict with accuracy and classification report as string.
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    print(f"Accuracy: {acc:.4f}\n")
    print(report)
    return {'accuracy': acc, 'report': report}


def plot_confusion_matrix(y_true, y_pred, labels: Optional[Sequence[str]] = None,
                          normalize: bool = False, figsize: Tuple[int,int]=(8,6),
                          cmap: str = 'Blues', save_path: Optional[str] = None) -> np.ndarray:
    """Plot and optionally save a confusion matrix heatmap. Returns the matrix (normalized if requested)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    plt.close()
    return cm


def plot_calibration_curve(estimator, X, y_true, n_bins: int = 10, figsize: Tuple[int,int]=(8,6),
                           save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Plot calibration curve (reliability diagram).

    estimator must implement predict_proba. If not available, try decision_function and convert with sigmoid.
    Returns (prob_true, prob_pred) arrays used to make the plot.
    """
    # try predict_proba
    try:
        y_prob = estimator.predict_proba(X)
        if y_prob.ndim == 2 and y_prob.shape[1] > 1:
            # assume multi-class; take max probability per sample
            prob_pos = np.max(y_prob, axis=1)
        else:
            prob_pos = y_prob.ravel()
    except Exception:
        # fallback to decision_function -> sigmoid
        try:
            scores = estimator.decision_function(X)
            # if multiclass, take max score
            if scores.ndim == 2:
                scores = np.max(scores, axis=1)
            prob_pos = 1 / (1 + np.exp(-scores))
        except Exception:
            raise ValueError('Estimator has neither predict_proba nor decision_function')

    prob_true, prob_pred = calibration_curve(y_true, prob_pos, n_bins=n_bins, strategy='uniform')

    plt.figure(figsize=figsize)
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0,1],[0,1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved calibration curve to {save_path}")
    plt.close()
    return prob_true, prob_pred


def permutation_importance_features(estimator, X, y, n_repeats: int = 10, random_state: int = 0,
                                    scoring: Optional[str] = 'accuracy') -> pd.DataFrame:
    """Compute permutation importance and return a sorted DataFrame of importances."""
    if hasattr(X, 'values'):
        X_arr = X.values
        feature_names = X.columns.tolist()
    else:
        X_arr = np.asarray(X)
        feature_names = [f'feature_{i}' for i in range(X_arr.shape[1])]

    result = permutation_importance(estimator, X_arr, y, n_repeats=n_repeats,
                                    random_state=random_state, scoring=scoring)
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return imp_df


def shap_explain(estimator, X, nsamples: int = 100, save_plot: Optional[str] = None):
    """Run SHAP explanations if shap is installed. Returns shap_values and explains summary plot.

    For tree-based models shap.TreeExplainer is faster; for others use KernelExplainer (slower).
    If shap is not installed, raises informative ImportError.
    """
    try:
        import shap
    except Exception as e:
        raise ImportError('shap is not installed. Install shap via `pip install shap` to use this function.')

    # Use a sample background to speed up
    if hasattr(X, 'values'):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)

    background = X_arr[np.random.choice(X_arr.shape[0], min(100, X_arr.shape[0]), replace=False)]

    try:
        explainer = shap.TreeExplainer(estimator)
    except Exception:
        explainer = shap.KernelExplainer(lambda z: estimator.predict_proba(z), background)

    shap_values = explainer.shap_values(X_arr[:nsamples])

    # summary plot
    try:
        shap.summary_plot(shap_values, X_arr[:nsamples], show=False)
        if save_plot:
            os.makedirs(os.path.dirname(save_plot) or '.', exist_ok=True)
            plt.savefig(save_plot, bbox_inches='tight')
            print(f"Saved SHAP summary plot to {save_plot}")
            plt.close()
    except Exception:
        # If plotting fails (matplotlib backend), ignore
        pass

    return shap_values


def partial_dependence(estimator, X, feature: Union[str, int], grid: Optional[np.ndarray] = None,
                       grid_points: int = 20) -> pd.DataFrame:
    """Compute a simple partial dependence for a single feature by averaging predictions.

    Returns a DataFrame with columns: feature_value, pdp (predicted average), count
    """
    if hasattr(X, 'values'):
        X_arr = X.values.copy()
        feature_names = X.columns.tolist()
    else:
        X_arr = np.asarray(X).copy()
        feature_names = [f'feature_{i}' for i in range(X_arr.shape[1])]

    if isinstance(feature, str):
        if feature not in feature_names:
            raise ValueError('feature not found in X')
        idx = feature_names.index(feature)
    else:
        idx = int(feature)

    col = X_arr[:, idx]
    if grid is None:
        grid = np.linspace(col.min(), col.max(), grid_points)

    pdp_list = []
    for val in grid:
        X_temp = X_arr.copy()
        X_temp[:, idx] = val
        # use predict_proba if possible, else predict
        try:
            probs = estimator.predict_proba(X_temp)
            if probs.ndim == 2 and probs.shape[1] > 1:
                avg = np.max(probs, axis=1).mean()
            else:
                avg = probs.ravel().mean()
        except Exception:
            preds = estimator.predict(X_temp)
            # if labels are strings, can't average; instead compute proportion of most common class
            if np.issubdtype(preds.dtype, np.number):
                avg = preds.mean()
            else:
                # proportion of the modal label
                vals, counts = np.unique(preds, return_counts=True)
                avg = counts.max() / counts.sum()
        pdp_list.append({'feature_value': val, 'pdp': avg})

    return pd.DataFrame(pdp_list)


if __name__ == '__main__':
    # Quick example - edit these paths to match your files
    MODEL_PATH = '../models/cart_model.pkl'  # Change to your model path
    DATA_PATH = '../data/train.csv'  # Change to your data path
    SCALER_PATH = None  # Set to scaler .pkl path if you have one, or None
    OUTPUT_DIR = 'error_analysis_output'
    SAMPLE_SIZE = 1000  # Number of samples to use for analysis (faster)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading model...')
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    # model_data could be dict with 'model' key or direct estimator
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
    else:
        model = model_data

    print('Loading data...')
    df = pd.read_csv(DATA_PATH)
    if 'Target' not in df.columns:
        raise ValueError('Data CSV must contain a Target column')

    X = df.drop(columns=['Target', 'Student_ID']) if 'Student_ID' in df.columns else df.drop(columns=['Target'])
    y = df['Target']

    if SCALER_PATH:
        print(f'Loading scaler from {SCALER_PATH}...')
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)

    # sample for speed
    if SAMPLE_SIZE and SAMPLE_SIZE < len(X):
        sel = np.random.choice(len(X), SAMPLE_SIZE, replace=False)
        Xp = X.iloc[sel]
        yp = y.iloc[sel]
    else:
        Xp = X
        yp = y

    print('Evaluating model...')
    y_pred = model.predict(Xp)
    evaluate_model(yp, y_pred, labels=np.unique(y))

    print('Saving confusion matrix...')
    plot_confusion_matrix(yp, y_pred, labels=np.unique(y), save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))

    print('Attempting permutation importance...')
    try:
        imp = permutation_importance_features(model, Xp, yp)
        imp.to_csv(os.path.join(OUTPUT_DIR, 'permutation_importance.csv'), index=False)
        print('Saved permutation importance')
    except Exception as e:
        print('Permutation importance failed:', e)

    print('Attempting SHAP explanation (optional, may be slow)...')
    try:
        shap_values = shap_explain(model, Xp, nsamples=min(100, len(Xp)), save_plot=os.path.join(OUTPUT_DIR, 'shap_summary.png'))
        print('SHAP explanation computed')
    except Exception as e:
        print('SHAP explanation skipped:', e)

    print('Done. Check the error_analysis_output/ folder for results.')
