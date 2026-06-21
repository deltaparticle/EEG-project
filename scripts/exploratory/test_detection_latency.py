import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from src import config
from src.preprocessing import load_and_preprocess_raw, estimate_baseline_variance, reject_artifact_windows
from src.features import extract_features_from_epoch
from src.model import evaluate_model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mne

def slice_multishift_epochs_from_raw(raw, shifts=[0, 100, 200, 300, 400, 500]):
    """
    Slices raw continuous data into positive (intent) and negative (non-intent) epochs.
    For positive epochs, we slice them for multiple backward shifts in time.
    For negative epochs, the data remains identical across all shifts.
    
    Returns:
      epochs_list: list of dicts mapping shift_ms -> 2D numpy array (shape: n_channels, n_times)
      labels_list: list of integers (1 for positive, 0 for negative)
      trial_ids_list: list of integers (trial grouping to prevent leakage)
    """
    sfreq = raw.info['sfreq']
    window_samples = int(config.WINDOW_DURATION * sfreq) # Window size in samples
    
    # Get EEG data
    eeg_data = raw.get_data(picks=config.CHANNELS_TO_USE)
    
    # Find events
    events = mne.find_events(raw, stim_channel="Status", verbose='WARNING')
    
    epochs_list = []
    labels_list = []
    trial_ids_list = []
    
    current_run = None
    trial_counter = 0
    
    for i in range(len(events)):
        sample_idx = events[i][0]
        code = events[i][2]
        
        # Get next event sample
        next_sample_idx = events[i+1][0] if i+1 < len(events) else eeg_data.shape[1]
        
        # Track run type
        if code in [config.RUN_PRONOUNCED, config.RUN_INNER, config.RUN_VISUALIZED]:
            current_run = code
            continue
            
        # Exclude covert runs
        if current_run == config.RUN_INNER:
            continue
            
        # Trial cue onset
        if code in [31, 32, 33, 34]:
            trial_counter += 1
            # Negative windows during cue
            for start_idx in range(sample_idx, next_sample_idx - window_samples, window_samples):
                epoch = eeg_data[:, start_idx:start_idx + window_samples]
                if epoch.shape[1] == window_samples:
                    epochs_list.append({s: epoch for s in shifts})
                    labels_list.append(0)
                    trial_ids_list.append(trial_counter)
            continue
            
        # Concentration onset
        if code == 42:
            # Negative windows during concentration
            for start_idx in range(sample_idx, next_sample_idx - window_samples, window_samples):
                epoch = eeg_data[:, start_idx:start_idx + window_samples]
                if epoch.shape[1] == window_samples:
                    epochs_list.append({s: epoch for s in shifts})
                    labels_list.append(0)
                    trial_ids_list.append(trial_counter)
            continue
            
        # Action onset
        if code == config.TRIGGER_ACTION_ONSET:
            if current_run == config.RUN_PRONOUNCED:
                # Slice positive epochs for each shift
                shift_dict = {}
                valid = True
                for s in shifts:
                    shift_samples = int((s / 1000.0) * sfreq)
                    start_idx = sample_idx - window_samples - shift_samples
                    end_idx = sample_idx - shift_samples
                    if start_idx >= 0:
                        epoch = eeg_data[:, start_idx:end_idx]
                        if epoch.shape[1] == window_samples:
                            shift_dict[s] = epoch
                        else:
                            valid = False
                    else:
                        valid = False
                
                if valid:
                    epochs_list.append(shift_dict)
                    labels_list.append(1)
                    trial_ids_list.append(trial_counter)
                    
            elif current_run == config.RUN_VISUALIZED:
                # Negative windows during visualization
                for start_idx in range(sample_idx, next_sample_idx - window_samples, window_samples):
                    epoch = eeg_data[:, start_idx:start_idx + window_samples]
                    if epoch.shape[1] == window_samples:
                        epochs_list.append({s: epoch for s in shifts})
                        labels_list.append(0)
                        trial_ids_list.append(trial_counter)
            continue
            
        # Rest onset
        if code == config.TRIGGER_REST_ONSET:
            # Negative windows during rest
            limit_idx = min(next_sample_idx, sample_idx + int(4.0 * sfreq))
            for start_idx in range(sample_idx, limit_idx - window_samples, window_samples):
                epoch = eeg_data[:, start_idx:start_idx + window_samples]
                if epoch.shape[1] == window_samples:
                    epochs_list.append({s: epoch for s in shifts})
                    labels_list.append(0)
                    trial_ids_list.append(trial_counter)
            continue
            
    return epochs_list, labels_list, trial_ids_list

def balance_indices(indices, y, ratio_n):
    """
    Selects indices to achieve a 1:N positive-to-negative class ratio.
    """
    split_y = y[indices]
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

def run_detection_latency_evaluation(subject, shifts=[0, 100, 200, 300, 400, 500]):
    print(f"\n==================================================")
    print(f"  RUNNING DETECTION LATENCY EVALUATION FOR: {subject}")
    print(f"==================================================")
    
    sessions = ["ses-01", "ses-02", "ses-03"]
    
    # Compute baseline variance thresholds
    print(f"[{subject}] Estimating baseline variance thresholds...")
    all_subj_variances = []
    
    for session in sessions:
        try:
            raw = load_and_preprocess_raw(subject, session)
            session_vars = estimate_baseline_variance(raw)
            if len(session_vars) > 0:
                all_subj_variances.append(session_vars)
        except Exception as e:
            print(f"  [{subject} | {session}] Error calculating baseline: {e}")
            
    if len(all_subj_variances) > 0:
        combined_vars = np.vstack(all_subj_variances)
        var_thresholds = np.percentile(combined_vars, 95, axis=0)
        print(f"  [{subject}] Baseline Variance Thresholds (95th percentile): "
              f"{list(np.round(var_thresholds*1e12, 2))} uV^2")
    else:
        var_thresholds = None
        print(f"  [{subject}] Warning: No baseline data found. Variance thresholding skipped.")
        
    # Extract raw epochs
    X_subj = []
    y_subj = []
    trial_ids_subj = []
    session_ids_subj = []
    trial_offset = 0
    
    for idx, session in enumerate(sessions, 1):
        try:
            raw = load_and_preprocess_raw(subject, session)
            
            # Slice positive/negative epochs with multishift
            epochs, labels, trial_ids = slice_multishift_epochs_from_raw(raw, shifts=shifts)
            if len(epochs) == 0:
                print(f"  [{subject} | {session}] Warning: No epochs found.")
                continue
                
            # Perform artifact rejection
            epochs_0 = np.array([ep[0] for ep in epochs])
            clean_indices = reject_artifact_windows(
                epochs_0, var_thresholds=var_thresholds, threshold_uv=config.ARTIFACT_THRESHOLD
            )
            
            clean_epochs = [epochs[i] for i in clean_indices]
            clean_labels = [labels[i] for i in clean_indices]
            clean_trial_ids = [trial_ids[i] for i in clean_indices]
            
            if len(clean_epochs) == 0:
                print(f"  [{subject} | {session}] Warning: No epochs left after artifact rejection.")
                continue
                
            X_subj.extend(clean_epochs)
            y_subj.extend(clean_labels)
            trial_ids_subj.extend(list(np.array(clean_trial_ids) + trial_offset))
            session_ids_subj.extend([idx] * len(clean_epochs))
            
            trial_offset = max(trial_ids_subj) if len(trial_ids_subj) > 0 else trial_offset
            
        except Exception as e:
            print(f"  [{subject} | {session}] Error processing session: {e}")
            
    if len(X_subj) == 0:
        print(f"Error: No data available for subject {subject}.")
        return []
        
    y_subj = np.array(y_subj)
    trial_ids_subj = np.array(trial_ids_subj)
    session_ids_subj = np.array(session_ids_subj)
    
    print(f"\n[{subject}] Combined dataset summary:")
    print(f"  Total Epochs: {len(X_subj)}")
    print(f"  Total Unique Trial IDs: {len(np.unique(trial_ids_subj))}")
    print(f"  Raw Class Distribution: Pos (Intent)={np.sum(y_subj == 1)}, Neg (Rest)={np.sum(y_subj == 0)}")
    
    subject_results = []
    
    # Loop over class ratios
    for ratio in [1, 2]:
        print(f"\n--- Evaluating Ratio 1:{ratio} (Randomized 70/15/15 trial split) ---")
        
        # Determine trial-level split (70/15/15)
        unique_trials = np.unique(trial_ids_subj)
        np.random.seed(42)
        np.random.shuffle(unique_trials)
        n_trials = len(unique_trials)
        n_train = int(0.70 * n_trials)
        n_val = int(0.15 * n_trials)
        
        train_trials = set(unique_trials[:n_train])
        val_trials = set(unique_trials[n_train:n_train+n_val])
        test_trials = set(unique_trials[n_train+n_val:])
        
        train_idx_all = [i for i, tid in enumerate(trial_ids_subj) if tid in train_trials]
        val_idx_all = [i for i, tid in enumerate(trial_ids_subj) if tid in val_trials]
        test_idx_all = [i for i, tid in enumerate(trial_ids_subj) if tid in test_trials]
        
        # Balance splits using deterministic seed 42
        np.random.seed(42)
        train_indices = balance_indices(train_idx_all, y_subj, ratio)
        val_indices = balance_indices(val_idx_all, y_subj, ratio)
        test_indices = balance_indices(test_idx_all, y_subj, ratio)
        
        # Train model on shift=0
        X_train_raw = np.array([X_subj[i][0] for i in train_indices])
        X_val_raw = np.array([X_subj[i][0] for i in val_indices])
        y_train = y_subj[train_indices]
        y_val = y_subj[val_indices]
        
        # Z-score normalization statistics
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
            
        # Extract features
        X_train_features = np.array([extract_features_from_epoch(epoch, use_18=True) for epoch in X_train_norm])
        X_val_features = np.array([extract_features_from_epoch(epoch, use_18=True) for epoch in X_val_norm])
        
        # Train Linear SVM
        X_combined = np.vstack([X_train_features, X_val_features])
        y_combined = np.concatenate([y_train, y_val])
        
        print(f"  Fitting Linear SVM classifier...")
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', C=2.0, class_weight='balanced', random_state=42))
        ])
        model.fit(X_combined, y_combined)
        
        # Evaluate on test set across shifts
        for shift in shifts:
            X_test_raw = np.array([X_subj[i][shift] for i in test_indices])
            y_test = y_subj[test_indices]
            
            # Z-score normalize test epochs
            X_test_norm = X_test_raw.copy()
            for ch in range(n_channels):
                X_test_norm[:, ch, :] = (X_test_raw[:, ch, :] - means[ch]) / (stds[ch] + 1e-8)
                
            # Extract features
            X_test_features = np.array([extract_features_from_epoch(epoch, use_18=True) for epoch in X_test_norm])
            
            metrics = evaluate_model(model, X_test_features, y_test)
            print(f"    Shift {shift:3d} ms: Recall (Detection Rate)={metrics['recall']:.2%}, F1={metrics['f1']:.2%}")
            
            subject_results.append({
                "Subject": subject,
                "Ratio": f"1:{ratio}",
                "Shift_ms": shift,
                "Accuracy": metrics["accuracy"],
                "Balanced_Accuracy": metrics["balanced_accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1_Score": metrics["f1"],
                "ROC_AUC": metrics["roc_auc"],
                "FPR": metrics["fpr"]
            })
            
    return subject_results

if __name__ == "__main__":
    import sys
    
    shifts = [0, 100, 200, 300, 400, 500]
    all_results = []
    
    for subj in config.SUBJECTS:
        subj_res = run_detection_latency_evaluation(subj, shifts=shifts)
        all_results.extend(subj_res)
        
    df_all = pd.DataFrame(all_results)
    
    # Save the detailed results to CSV
    csv_path = r"d:\Temple Project\detection_latency_results.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\nSaved detailed latency evaluation results to: {csv_path}")
    
    # Print summary table
    print("\n" + "="*70)
    print("  DETECTION LATENCY PROFILE SUMMARY (AVERAGE ACROSS SUBJECTS)")
    print("="*70)
    
    summary = df_all.groupby(["Ratio", "Shift_ms"])[["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "FPR"]].mean().reset_index()
    
    print(summary.to_string(index=False, formatters={
        "Accuracy": "{:.2%}".format,
        "Balanced_Accuracy": "{:.2%}".format,
        "Precision": "{:.2%}".format,
        "Recall": "{:.2%}".format,
        "F1_Score": "{:.2%}".format,
        "ROC_AUC": "{:.2%}".format,
        "FPR": "{:.2%}".format
    }))
