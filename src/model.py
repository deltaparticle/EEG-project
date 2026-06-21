import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def train_svm_with_tuning(X_train, y_train, X_val, y_val, subject=None, random_seed=42):
    """
    Trains a Linear SVM classifier on the combined train and validation splits.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    
    # Combine train and validation splits
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    # Instantiate and fit the pipeline
    print(f"  [Model Training] Fitting unified Linear SVM (C=0.5) classifier...")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='linear', C=0.5, class_weight='balanced', random_state=random_seed))
    ])

    pipeline.fit(X_combined, y_combined)
    return pipeline

def evaluate_model(clf, X, y):
    """
    Evaluates the trained SVM classifier and computes metrics.
    """
    y_pred = clf.predict(X)
    
    # Get probability estimates or decision function values
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_prob = clf.decision_function(X)
    else:
        y_prob = None
        
    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    try:
        if y_prob is not None:
            auc = roc_auc_score(y, y_prob)
        else:
            auc = 0.5
    except Exception:
        auc = 0.5
        
    # Calculate false positive rate
    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        fpr = 0.0
        cm = [[0, 0], [0, 0]]
        
    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "fpr": fpr,
        "confusion_matrix": cm
    }
