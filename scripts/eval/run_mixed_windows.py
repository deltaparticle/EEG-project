import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from src import config
from src.preprocessing import load_and_preprocess_raw, estimate_baseline_variance, reject_artifact_windows
from src.features import slice_epochs_from_raw, extract_features_from_epoch
from src.model import evaluate_model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def run_mixed_window_pipeline(subject):
    print(f"\n==================================================")
    print(f"  RUNNING PIPELINE FOR SUBJECT: {subject}")
    print(f"==================================================")
    
    percentile = 95
    sessions = ["ses-01", "ses-02", "ses-03"]
    
    # Compute baseline variance thresholds
    all_subj_variances = []
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
        var_thresholds = np.percentile(combined_vars, percentile, axis=0)
    else:
        var_thresholds = None
        
    # Slice standard epochs
    X_subj = []
    y_subj = []
    trial_ids_subj = []
    session_ids_subj = []
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
                
            X_subj.append(np.array(clean_epochs))
            y_subj.append(np.array(clean_labels))
            trial_ids_subj.append(np.array(clean_trial_ids) + trial_offset)
            session_ids_subj.append(np.array([idx] * len(clean_epochs)))
            
            trial_offset = max(trial_ids_subj[-1]) if len(clean_trial_ids) > 0 else trial_offset
        except Exception as e:
            print(f"  [{subject} | {session}] Error processing: {e}")
            
    if len(X_subj) == 0:
        return []
        
    X = np.vstack(X_subj) # shape: (n_epochs, n_channels, 125)
    y = np.concatenate(y_subj)
    trial_ids = np.concatenate(trial_ids_subj)
    session_ids = np.concatenate(session_ids_subj)
    
    results = []
    
    # Balance dataset to 1:N ratio
    def balance_indices(indices, y_labels, ratio_n):
        split_y = y_labels[indices]
        pos_idx = np.where(split_y == 1)[0]
        neg_idx = np.where(split_y == 0)[0]
        n_pos = len(pos_idx)
        n_neg_needed = int(n_pos * ratio_n)
        
        if len(neg_idx) < n_neg_needed:
            sampled_neg_idx = neg_idx
        else:
            sampled_neg_idx = np.random.choice(neg_idx, size=n_neg_needed, replace=False)
            
        final_idx = np.concatenate([pos_idx, sampled_neg_idx])
        np.random.shuffle(final_idx)
        return [indices[i] for i in final_idx]
        
    # Loop over class ratios
    for ratio in [1, 2]:
        unique_trials = np.unique(trial_ids)
        np.random.seed(42)
        np.random.shuffle(unique_trials)
        n_trials = len(unique_trials)
        n_train = int(0.70 * n_trials)
        n_val = int(0.15 * n_trials)
        
        train_trials = set(unique_trials[:n_train])
        val_trials = set(unique_trials[n_train:n_train+n_val])
        test_trials = set(unique_trials[n_train+n_val:])
        
        train_idx_all = [i for i, tid in enumerate(trial_ids) if tid in train_trials]
        val_idx_all = [i for i, tid in enumerate(trial_ids) if tid in val_trials]
        test_idx_all = [i for i, tid in enumerate(trial_ids) if tid in test_trials]
        
        np.random.seed(42)
        train_indices = balance_indices(train_idx_all, y, ratio)
        val_indices = balance_indices(val_idx_all, y, ratio)
        test_indices = balance_indices(test_idx_all, y, ratio)
        
        X_train_raw = X[train_indices] # shape: (n_train, 4, 125)
        X_val_raw = X[val_indices]     # shape: (n_val, 4, 125)
        X_test_raw = X[test_indices]   # shape: (n_test, 4, 125)
        
        y_train = y[train_indices]
        y_val = y[val_indices]
        y_test = y[test_indices]
        
        # Fit channel-wise Z-score statistics on training epochs
        n_channels = X_train_raw.shape[1]
        means = np.zeros(n_channels)
        stds = np.ones(n_channels)
        for ch in range(n_channels):
            ch_data = X_train_raw[:, ch, :]
            means[ch] = np.mean(ch_data)
            stds[ch] = np.std(ch_data)
            
        # Z-score normalize train & val
        X_train_norm = X_train_raw.copy()
        X_val_norm = X_val_raw.copy()
        for ch in range(n_channels):
            X_train_norm[:, ch, :] = (X_train_raw[:, ch, :] - means[ch]) / (stds[ch] + 1e-8)
            X_val_norm[:, ch, :] = (X_val_raw[:, ch, :] - means[ch]) / (stds[ch] + 1e-8)
            
        # Extract features for train & val
        X_train_features = np.array([extract_features_from_epoch(epoch, use_18=True) for epoch in X_train_norm])
        X_val_features = np.array([extract_features_from_epoch(epoch, use_18=True) for epoch in X_val_norm])
        
        # Train model on combined train & val
        X_combined = np.vstack([X_train_features, X_val_features])
        y_combined = np.concatenate([y_train, y_val])
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', C=2.0, class_weight='balanced', random_state=42))
        ])
        model.fit(X_combined, y_combined)
        
        # Evaluate on test set with custom window durations
        durations = {
            500: 125, # last 500 ms (full epoch)
            300: 75,  # last 300 ms
            200: 50   # last 200 ms
        }
        
        for dur, samples in durations.items():
            # Crop test raw epochs to target duration
            X_test_cropped = X_test_raw[:, :, -samples:] # shape: (n_test, 4, samples)
            
            # Normalize cropped test epochs
            X_test_norm = X_test_cropped.copy()
            for ch in range(n_channels):
                X_test_norm[:, ch, :] = (X_test_cropped[:, ch, :] - means[ch]) / (stds[ch] + 1e-8)
                
            # Extract features from normalized test epochs
            X_test_features = np.array([extract_features_from_epoch(epoch, use_18=True) for epoch in X_test_norm])
            
            metrics = evaluate_model(model, X_test_features, y_test)
            
            results.append({
                "Subject": subject,
                "Ratio": f"1:{ratio}",
                "Test_Window_ms": dur,
                "Accuracy": metrics["accuracy"],
                "Balanced_Accuracy": metrics["balanced_accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1_Score": metrics["f1"],
                "ROC_AUC": metrics["roc_auc"],
                "FPR": metrics["fpr"]
            })
            
    return results

if __name__ == "__main__":
    import sys
    
    all_results = []
    
    for subj in config.SUBJECTS:
        res = run_mixed_window_pipeline(subj)
        all_results.extend(res)
        
    df = pd.DataFrame(all_results)
    csv_path = r"d:\Temple Project\mixed_windows_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved mixed window evaluation results to: {csv_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("  MIXED WINDOWS EVALUATION SUMMARY (TRAINED ON 500ms, TESTED ON SHORTER WINDOWS)")
    print("="*80)
    
    summary = df.groupby(["Test_Window_ms", "Ratio"])[["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "FPR"]].mean().reset_index()
    print(summary.to_string(index=False, formatters={
        "Accuracy": "{:.2%}".format,
        "Balanced_Accuracy": "{:.2%}".format,
        "Precision": "{:.2%}".format,
        "Recall": "{:.2%}".format,
        "F1_Score": "{:.2%}".format,
        "ROC_AUC": "{:.2%}".format,
        "FPR": "{:.2%}".format
    }))
