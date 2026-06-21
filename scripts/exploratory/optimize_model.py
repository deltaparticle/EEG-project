import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sys
import numpy as np
import pandas as pd
from src import config
from src.preprocessing import load_and_preprocess_raw, estimate_baseline_variance, reject_artifact_windows
from src.features import slice_epochs_from_raw, extract_features_from_epoch
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def load_clean_subject_data(subject):
    sessions = ["ses-01", "ses-02", "ses-03"]
    all_subj_variances = []
    
    for session in sessions:
        try:
            raw = load_and_preprocess_raw(subject, session)
            session_vars = estimate_baseline_variance(raw)
            if len(session_vars) > 0:
                all_subj_variances.append(session_vars)
        except Exception as e:
            pass
            
    if len(all_subj_variances) > 0:
        combined_vars = np.vstack(all_subj_variances)
        var_thresholds = np.percentile(combined_vars, 95, axis=0)
    else:
        var_thresholds = None
        
    X_list = []
    y_list = []
    trial_ids_list = []
    trial_offset = 0
    
    for idx, session in enumerate(sessions, 1):
        try:
            raw = load_and_preprocess_raw(subject, session)
            epochs, labels, trial_ids = slice_epochs_from_raw(raw)
            if len(epochs) == 0:
                continue
                
            epochs_array = np.array(epochs)
            clean_indices = reject_artifact_windows(
                epochs_array, var_thresholds=var_thresholds, threshold_uv=config.ARTIFACT_THRESHOLD
            )
            
            clean_epochs = [epochs[i] for i in clean_indices]
            clean_labels = [labels[i] for i in clean_indices]
            clean_trial_ids = [trial_ids[i] for i in clean_indices]
            
            if len(clean_epochs) == 0:
                continue
                
            X_list.append(np.array(clean_epochs))
            y_list.append(np.array(clean_labels))
            trial_ids_list.append(np.array(clean_trial_ids) + trial_offset)
            trial_offset = max(trial_ids_list[-1]) if len(clean_trial_ids) > 0 else trial_offset
        except Exception as e:
            pass
            
    if len(X_list) == 0:
        return None
        
    X_clean = np.vstack(X_list)
    y_clean = np.concatenate(y_list)
    trial_ids = np.concatenate(trial_ids_list)
    return X_clean, y_clean, trial_ids

def balance_dataset(X, y, ratio, seed=42):
    np.random.seed(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg_needed = int(n_pos * ratio)
    
    if len(neg_idx) < n_neg_needed:
        sampled_neg_idx = neg_idx
    else:
        sampled_neg_idx = np.random.choice(neg_idx, size=n_neg_needed, replace=False)
        
    final_idx = np.concatenate([pos_idx, sampled_neg_idx])
    np.random.shuffle(final_idx)
    return X[final_idx], y[final_idx]

def extract_subject_features(X_raw, means=None, stds=None):
    n_channels = X_raw.shape[1]
    if means is None or stds is None:
        means = np.zeros(n_channels)
        stds = np.ones(n_channels)
        for ch in range(n_channels):
            ch_data = X_raw[:, ch, :]
            means[ch] = np.mean(ch_data)
            stds[ch] = np.std(ch_data)
            
    X_norm = X_raw.copy()
    for ch in range(n_channels):
        X_norm[:, ch, :] = (X_raw[:, ch, :] - means[ch]) / (stds[ch] + 1e-8)
        
    features = np.array([extract_features_from_epoch(epoch, use_18=True) for epoch in X_norm])
    return features, means, stds

if __name__ == "__main__":
    print("==================================================")
    print("  LOADING AND PROCESSING DATA FOR SVM OPTIMIZATION")
    print("==================================================")
    
    subject_data = {}
    for subj in config.SUBJECTS:
        res = load_clean_subject_data(subj)
        if res is not None:
            subject_data[subj] = res
            print(f"  {subj} loaded: {res[0].shape[0]} epochs")
            
    models = {
        "Linear SVM (C=2.0)": SVC(kernel="linear", C=2.0, class_weight="balanced", random_state=42),
        "Linear SVM (C=0.5)": SVC(kernel="linear", C=0.5, class_weight="balanced", random_state=42),
        "RBF SVM (C=1.0)": SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42),
        "Sparse L1 SVM (C=0.5)": LinearSVC(penalty="l1", C=0.5, loss="squared_hinge", dual=False, class_weight="balanced", random_state=42)
    }
    
    ratio = 1 # 1:1 ratio for benchmarking
    results = []
    
    for name, clf in models.items():
        print(f"\nEvaluating: {name}...")
        
        # Within-subject evaluation
        within_accs = []
        for subj, (X, y, trial_ids) in subject_data.items():
            unique_trials = np.unique(trial_ids)
            np.random.seed(42)
            np.random.shuffle(unique_trials)
            n_trials = len(unique_trials)
            n_train = int(0.70 * n_trials)
            n_val = int(0.15 * n_trials)
            
            train_trials = set(unique_trials[:n_train])
            val_trials = set(unique_trials[n_train:n_train+n_val])
            test_trials = set(unique_trials[n_train+n_val:])
            
            train_idx = [i for i, tid in enumerate(trial_ids) if tid in train_trials]
            val_idx = [i for i, tid in enumerate(trial_ids) if tid in val_trials]
            test_idx = [i for i, tid in enumerate(trial_ids) if tid in test_trials]
            
            # Balance splits
            train_indices = balance_dataset(X[train_idx], y[train_idx], ratio, seed=42)
            val_indices = balance_dataset(X[val_idx], y[val_idx], ratio, seed=42)
            test_indices = balance_dataset(X[test_idx], y[test_idx], ratio, seed=42)
            
            X_train_raw, y_train = train_indices
            X_val_raw, y_val = val_indices
            X_test_raw, y_test = test_indices
            
            X_combined_raw = np.vstack([X_train_raw, X_val_raw])
            y_combined = np.concatenate([y_train, y_val])
            
            # Extract features using training normalization statistics
            X_combined_feats, means, stds = extract_subject_features(X_combined_raw)
            X_test_feats, _, _ = extract_subject_features(X_test_raw, means, stds)
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', clf)
            ])
            pipeline.fit(X_combined_feats, y_combined)
            preds = pipeline.predict(X_test_feats)
            acc = accuracy_score(y_test, preds)
            within_accs.append(acc)
            
        avg_within_acc = np.mean(within_accs)
        
        # Cross-subject transfer evaluation
        cross_accs = []
        for src_subj in config.SUBJECTS:
            if src_subj not in subject_data:
                continue
            X_src, y_src, _ = subject_data[src_subj]
            X_train_raw, y_train = balance_dataset(X_src, y_src, ratio, seed=42)
            X_train_feats, means_src, stds_src = extract_subject_features(X_train_raw)
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', clf)
            ])
            pipeline.fit(X_train_feats, y_train)
            
            for target_subj in config.SUBJECTS:
                if target_subj == src_subj or target_subj not in subject_data:
                    continue
                X_tgt, y_tgt, _ = subject_data[target_subj]
                X_test_raw, y_test = balance_dataset(X_tgt, y_tgt, ratio, seed=42)
                X_test_feats, _, _ = extract_subject_features(X_test_raw) # Z-score target using own statistics
                
                preds = pipeline.predict(X_test_feats)
                acc = accuracy_score(y_test, preds)
                cross_accs.append(acc)
                
        avg_cross_acc = np.mean(cross_accs)
        
        print(f"  Avg Within-Subject Accuracy: {avg_within_acc:.2%}")
        print(f"  Avg Cross-Subject Accuracy:  {avg_cross_acc:.2%}")
        
        results.append({
            "Model": name,
            "Within_Subject_Accuracy": avg_within_acc,
            "Cross_Subject_Accuracy": avg_cross_acc
        })
        
    df_results = pd.DataFrame(results)
    print("\n" + "="*60)
    print("  SVM ARCHITECTURE SELECTION COMPARISON")
    print("="*60)
    print(df_results.to_string(index=False, formatters={
        "Within_Subject_Accuracy": "{:.2%}".format,
        "Cross_Subject_Accuracy": "{:.2%}".format
    }))
