import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sys
import numpy as np
import pandas as pd
from src import config
from src.preprocessing import load_and_preprocess_raw, estimate_baseline_variance, reject_artifact_windows
from src.features import slice_epochs_from_raw, extract_features_from_epoch
from src.model import evaluate_model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_clean_subject_data(subject):
    """
    Loads raw BDF files for a subject, computes baseline thresholds,
    slices 500 ms epochs, and applies artifact rejection.
    Returns:
      X_clean (n_epochs, 4, 125)
      y_clean (n_epochs,)
      trial_ids (n_epochs,)
    """
    sessions = ["ses-01", "ses-02", "ses-03"]
    all_subj_variances = []
    
    # Estimate baseline variance thresholds
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
        
    # Slice epochs
    X_list = []
    y_list = []
    trial_ids_list = []
    trial_offset = 0
    
    for session in sessions:
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

def balance_dataset(X, y, trial_ids, ratio, seed=42):
    """
    Balances dataset to 1:Ratio positive-to-negative epochs.
    """
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
    return X[final_idx], y[final_idx], trial_ids[final_idx]

if __name__ == "__main__":
    print("==================================================")
    print("  LOADING AND PREPROCESSING CLEAN DATA FOR ALL 10 SUBJECTS")
    print("==================================================")
    
    pooled_X = []
    pooled_y = []
    pooled_global_trial_ids = []
    
    for subj in config.SUBJECTS:
        print(f"\nProcessing {subj}...")
        res = load_clean_subject_data(subj)
        if res is not None:
            X, y, trial_ids = res
            
            # Z-score normalize channels independently
            n_channels = X.shape[1]
            means = np.zeros(n_channels)
            stds = np.ones(n_channels)
            for ch in range(n_channels):
                ch_data = X[:, ch, :]
                means[ch] = np.mean(ch_data)
                stds[ch] = np.std(ch_data)
                
            X_norm = X.copy()
            for ch in range(n_channels):
                X_norm[:, ch, :] = (X[:, ch, :] - means[ch]) / (stds[ch] + 1e-8)
                
            # Create globally unique trial IDs
            global_trials = [f"{subj}_{tid}" for tid in trial_ids]
            
            pooled_X.append(X_norm)
            pooled_y.append(y)
            pooled_global_trial_ids.extend(global_trials)
            print(f"  {subj} loaded: {X.shape[0]} clean & normalized epochs")
            
    if len(pooled_X) == 0:
        print("Error: No subject data loaded. Exiting.")
        sys.exit(1)
        
    # Combine lists into arrays
    X_all = np.vstack(pooled_X) # shape: (total_epochs, 4, 125)
    y_all = np.concatenate(pooled_y) # shape: (total_epochs,)
    global_trial_ids_all = np.array(pooled_global_trial_ids) # shape: (total_epochs,)
    
    print("\n" + "="*50)
    print("  POOLED MULTI-SUBJECT DATASET SUMMARY")
    print("="*50)
    print(f"  Total Clean Epochs: {X_all.shape[0]}")
    print(f"  Total Channels:     {X_all.shape[1]}")
    print(f"  Total Time Points:  {X_all.shape[2]}")
    print(f"  Class Distribution: Pos (Intent)={np.sum(y_all == 1)}, Neg (Rest)={np.sum(y_all == 0)}")
    
    results = []
    
    # Loop over class ratios
    for ratio in [1, 2]:
        print(f"\n--- Training Unified Model: Class Ratio 1:{ratio} ---")
        
        # Trial-level randomized split on pooled dataset (70/15/15)
        unique_global_trials = np.unique(global_trial_ids_all)
        np.random.seed(42)
        np.random.shuffle(unique_global_trials)
        n_trials = len(unique_global_trials)
        n_train = int(0.70 * n_trials)
        n_val = int(0.15 * n_trials)
        
        train_trials = set(unique_global_trials[:n_train])
        val_trials = set(unique_global_trials[n_train:n_train+n_val])
        test_trials = set(unique_global_trials[n_train+n_val:])
        
        train_idx_all = [i for i, gtid in enumerate(global_trial_ids_all) if gtid in train_trials]
        val_idx_all = [i for i, gtid in enumerate(global_trial_ids_all) if gtid in val_trials]
        test_idx_all = [i for i, gtid in enumerate(global_trial_ids_all) if gtid in test_trials]
        
        # Balance each split
        X_train_raw, y_train, _ = balance_dataset(X_all[train_idx_all], y_all[train_idx_all], global_trial_ids_all[train_idx_all], ratio, seed=42)
        X_val_raw, y_val, _ = balance_dataset(X_all[val_idx_all], y_all[val_idx_all], global_trial_ids_all[val_idx_all], ratio, seed=42)
        X_test_raw, y_test, _ = balance_dataset(X_all[test_idx_all], y_all[test_idx_all], global_trial_ids_all[test_idx_all], ratio, seed=42)
        
        print(f"  Class counts (1:{ratio} target ratio):")
        print(f"    Train: {len(y_train)} total (Pos: {np.sum(y_train == 1)}, Neg: {np.sum(y_train == 0)})")
        print(f"    Val:   {len(y_val)} total (Pos: {np.sum(y_val == 1)}, Neg: {np.sum(y_val == 0)})")
        print(f"    Test:  {len(y_test)} total (Pos: {np.sum(y_test == 1)}, Neg: {np.sum(y_test == 0)})")
        
        # Extract features
        print("  Extracting PSD and temporal features...")
        X_train_features = np.array([extract_features_from_epoch(epoch, use_18=True) for epoch in X_train_raw])
        X_val_features = np.array([extract_features_from_epoch(epoch, use_18=True) for epoch in X_val_raw])
        X_test_features = np.array([extract_features_from_epoch(epoch, use_18=True) for epoch in X_test_raw])
        
        # Train Linear SVM on combined train and validation sets
        X_combined = np.vstack([X_train_features, X_val_features])
        y_combined = np.concatenate([y_train, y_val])
        
        print("  Training Linear SVM (C=0.5) classifier...")
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', C=0.5, class_weight='balanced', random_state=42))
        ])
        model.fit(X_combined, y_combined)
        
        # Evaluate on the unseen Test set
        print("  Evaluating on the Test set...")
        metrics = evaluate_model(model, X_test_features, y_test)
        
        results.append({
            "Ratio": f"1:{ratio}",
            "Accuracy": metrics["accuracy"],
            "Balanced_Accuracy": metrics["balanced_accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1_Score": metrics["f1"],
            "ROC_AUC": metrics["roc_auc"],
            "FPR": metrics["fpr"]
        })
        
    df_res = pd.DataFrame(results)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    csv_path = os.path.join(root_dir, "results", "unified_model_results.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved unified model results to: {csv_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("  UNIFIED MULTI-SUBJECT MODEL PERFORMANCE")
    print("="*80)
    print(df_res.to_string(index=False, formatters={
        "Accuracy": "{:.2%}".format,
        "Balanced_Accuracy": "{:.2%}".format,
        "Precision": "{:.2%}".format,
        "Recall": "{:.2%}".format,
        "F1_Score": "{:.2%}".format,
        "ROC_AUC": "{:.2%}".format,
        "FPR": "{:.2%}".format
    }))
