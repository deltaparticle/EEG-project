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
            print(f"  [{subject} | {session}] Error baseline: {e}")
            
    if len(all_subj_variances) > 0:
        combined_vars = np.vstack(all_subj_variances)
        var_thresholds = np.percentile(combined_vars, 95, axis=0)
    else:
        var_thresholds = None
        
    # Slice epochs
    X_list = []
    y_list = []
    
    for session in sessions:
        try:
            raw = load_and_preprocess_raw(subject, session)
            epochs, labels, _ = slice_epochs_from_raw(raw)
            if len(epochs) == 0:
                continue
                
            epochs_array = np.array(epochs)
            clean_indices = reject_artifact_windows(
                epochs_array, var_thresholds=var_thresholds, threshold_uv=config.ARTIFACT_THRESHOLD
            )
            
            clean_epochs = [epochs[i] for i in clean_indices]
            clean_labels = [labels[i] for i in clean_indices]
            
            if len(clean_epochs) == 0:
                continue
                
            X_list.append(np.array(clean_epochs))
            y_list.append(np.array(clean_labels))
        except Exception as e:
            print(f"  [{subject} | {session}] Error processing: {e}")
            
    if len(X_list) == 0:
        return np.empty((0, len(config.CHANNELS_TO_USE), 125)), np.empty((0,))
        
    X_clean = np.vstack(X_list)
    y_clean = np.concatenate(y_list)
    return X_clean, y_clean

def balance_dataset(X, y, ratio, seed=42):
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
    return X[final_idx], y[final_idx]

if __name__ == "__main__":
    # Load and process clean data for all subjects
    print("==================================================")
    print("  LOADING AND PREPROCESSING CLEAN DATA FOR ALL SUBJECTS")
    print("==================================================")
    
    subject_cache = {}
    for subj in config.SUBJECTS:
        print(f"\nProcessing {subj}...")
        X, y = load_clean_subject_data(subj)
        if len(X) > 0:
            subject_cache[subj] = (X, y)
            print(f"  {subj} loaded: {X.shape[0]} epochs (Pos: {np.sum(y==1)}, Neg: {np.sum(y==0)})")
            
    results = []
    
    # Loop over class ratios
    for ratio in [1, 2]:
        print(f"\n==================================================")
        print(f"  ALL-TO-ALL CROSS-SUBJECT TRANSFER (Ratio 1:{ratio})")
        print(f"==================================================")
        
        for src_subj in config.SUBJECTS:
            if src_subj not in subject_cache:
                continue
                
            print(f"\nTraining on {src_subj}...")
            X_src, y_src = subject_cache[src_subj]
            X_train_raw, y_train = balance_dataset(X_src, y_src, ratio, seed=42)
            
            # Z-score normalization statistics
            n_channels = X_train_raw.shape[1]
            means_src = np.zeros(n_channels)
            stds_src = np.ones(n_channels)
            for ch in range(n_channels):
                ch_data = X_train_raw[:, ch, :]
                means_src[ch] = np.mean(ch_data)
                stds_src[ch] = np.std(ch_data)
                
            # Z-score normalize raw data
            X_train_norm = X_train_raw.copy()
            for ch in range(n_channels):
                X_train_norm[:, ch, :] = (X_train_raw[:, ch, :] - means_src[ch]) / (stds_src[ch] + 1e-8)
                
            # Extract features
            X_train_features = np.array([extract_features_from_epoch(epoch, use_18=True) for epoch in X_train_norm])
            
            # Train model
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(kernel='linear', C=0.5, class_weight='balanced', random_state=42))
            ])
 
            model.fit(X_train_features, y_train)
            
            # Evaluate on all subjects
            for target_subj in config.SUBJECTS:
                if target_subj not in subject_cache:
                    continue
                    
                X_target, y_target = subject_cache[target_subj]
                
                # Balance target dataset
                X_test_raw, y_test = balance_dataset(X_target, y_target, ratio, seed=42)
                
                # Z-score normalize target data using target statistics
                means_target = np.zeros(n_channels)
                stds_target = np.ones(n_channels)
                for ch in range(n_channels):
                    ch_data = X_test_raw[:, ch, :]
                    means_target[ch] = np.mean(ch_data)
                    stds_target[ch] = np.std(ch_data)
                    
                X_test_norm = X_test_raw.copy()
                for ch in range(n_channels):
                    X_test_norm[:, ch, :] = (X_test_raw[:, ch, :] - means_target[ch]) / (stds_target[ch] + 1e-8)
                    
                # Extract features
                X_test_features = np.array([extract_features_from_epoch(epoch, use_18=True) for epoch in X_test_norm])
                
                metrics = evaluate_model(model, X_test_features, y_test)
                
                results.append({
                    "Train_Subject": src_subj,
                    "Test_Subject": target_subj,
                    "Ratio": f"1:{ratio}",
                    "Accuracy": metrics["accuracy"],
                    "Balanced_Accuracy": metrics["balanced_accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1_Score": metrics["f1"],
                    "ROC_AUC": metrics["roc_auc"],
                    "FPR": metrics["fpr"]
                })
                
                print(f"    Tested on {target_subj:6s}: F1_Score={metrics['f1']:.2%}")
                
    # Save cross-subject results
    df_res = pd.DataFrame(results)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    csv_path = os.path.join(root_dir, "results", "cross_subject_results.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved all-to-all cross-subject evaluation results to: {csv_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("  ALL-TO-ALL CROSS-SUBJECT PERFORMANCE SUMMARY")
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
