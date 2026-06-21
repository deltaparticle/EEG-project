import os
import sys
import numpy as np
import pandas as pd

# Add workspace root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.preprocessing import load_and_preprocess_raw, reject_artifact_windows
from src.features import extract_features_from_epoch
from src.model import evaluate_model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mne

def estimate_baseline_variance_dynamic(raw, window_duration_sec):
    sfreq = raw.info['sfreq']
    window_samples = int(window_duration_sec * sfreq)
    events = mne.find_events(raw, stim_channel="Status", verbose='WARNING')
    eeg_data = raw.get_data(picks=config.CHANNELS_TO_USE)
    
    variances = []
    for i in range(len(events)):
        sample_idx = events[i][0]
        code = events[i][2]
        if code == 13:
            next_sample = eeg_data.shape[1]
            if i + 1 < len(events) and events[i+1][2] == 14:
                next_sample = events[i+1][0]
            for start_idx in range(sample_idx, next_sample - window_samples, window_samples):
                epoch = eeg_data[:, start_idx:start_idx + window_samples]
                if epoch.shape[1] == window_samples:
                    epoch_var = np.var(epoch, axis=1)
                    variances.append(epoch_var)
    return np.array(variances) if len(variances) > 0 else np.empty((0, len(config.CHANNELS_TO_USE)))

def slice_epochs_from_raw_dynamic(raw, window_duration_sec):
    sfreq = raw.info['sfreq']
    window_samples = int(window_duration_sec * sfreq)
    eeg_data = raw.get_data(picks=config.CHANNELS_TO_USE)
    events = mne.find_events(raw, stim_channel="Status", verbose='WARNING')
    
    epochs_list = []
    labels_list = []
    trial_ids_list = []
    
    current_run = None
    trial_counter = 0
    
    for i in range(len(events)):
        sample_idx = events[i][0]
        code = events[i][2]
        next_sample_idx = events[i+1][0] if i+1 < len(events) else eeg_data.shape[1]
        
        if code in [config.RUN_PRONOUNCED, config.RUN_INNER, config.RUN_VISUALIZED]:
            current_run = code
            continue
            
        if current_run == config.RUN_INNER:
            continue
            
        if code in [31, 32, 33, 34]:
            trial_counter += 1
            for start_idx in range(sample_idx, next_sample_idx - window_samples, window_samples):
                epoch = eeg_data[:, start_idx:start_idx + window_samples]
                if epoch.shape[1] == window_samples:
                    epochs_list.append(epoch)
                    labels_list.append(0)
                    trial_ids_list.append(trial_counter)
            continue
            
        if code == 42:
            for start_idx in range(sample_idx, next_sample_idx - window_samples, window_samples):
                epoch = eeg_data[:, start_idx:start_idx + window_samples]
                if epoch.shape[1] == window_samples:
                    epochs_list.append(epoch)
                    labels_list.append(0)
                    trial_ids_list.append(trial_counter)
            continue
            
        if code == config.TRIGGER_ACTION_ONSET:
            if current_run == config.RUN_PRONOUNCED:
                start_idx = sample_idx - window_samples
                if start_idx >= 0:
                    epoch = eeg_data[:, start_idx:sample_idx]
                    if epoch.shape[1] == window_samples:
                        epochs_list.append(epoch)
                        labels_list.append(1)
                        trial_ids_list.append(trial_counter)
            elif current_run == config.RUN_VISUALIZED:
                for start_idx in range(sample_idx, next_sample_idx - window_samples, window_samples):
                    epoch = eeg_data[:, start_idx:start_idx + window_samples]
                    if epoch.shape[1] == window_samples:
                        epochs_list.append(epoch)
                        labels_list.append(0)
                        trial_ids_list.append(trial_counter)
            continue
            
        if code == config.TRIGGER_REST_ONSET:
            limit_idx = min(next_sample_idx, sample_idx + int(4.0 * sfreq))
            for start_idx in range(sample_idx, limit_idx - window_samples, window_samples):
                epoch = eeg_data[:, start_idx:start_idx + window_samples]
                if epoch.shape[1] == window_samples:
                    epochs_list.append(epoch)
                    labels_list.append(0)
                    trial_ids_list.append(trial_counter)
            continue
            
    return epochs_list, labels_list, trial_ids_list

def balance_indices(indices, y, ratio_n):
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

def get_processed_data(subject, window_sec, percentile=95):
    sessions = ["ses-01", "ses-02", "ses-03"]
    all_subj_variances = []
    for session in sessions:
        try:
            raw = load_and_preprocess_raw(subject, session)
            session_vars = estimate_baseline_variance_dynamic(raw, window_sec)
            if len(session_vars) > 0:
                all_subj_variances.append(session_vars)
        except Exception as e:
            pass
            
    if len(all_subj_variances) > 0:
        combined_vars = np.vstack(all_subj_variances)
        var_thresholds = np.percentile(combined_vars, percentile, axis=0)
    else:
        var_thresholds = None
        
    X_subj, y_subj, trial_ids_subj = [], [], []
    trial_offset = 0
    
    for session in sessions:
        try:
            raw = load_and_preprocess_raw(subject, session)
            epochs, labels, trial_ids = slice_epochs_from_raw_dynamic(raw, window_sec)
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
            trial_offset = max(trial_ids_subj[-1]) if len(clean_trial_ids) > 0 else trial_offset
        except Exception as e:
            pass
            
    if len(X_subj) == 0:
        return None, None, None
        
    return np.vstack(X_subj), np.concatenate(y_subj), np.concatenate(trial_ids_subj)

def run_experiment_for_subject(subject):
    print(f"\nEvaluating subject {subject}...")
    
    # Load 500 ms, 300 ms, 200 ms data
    data_500 = get_processed_data(subject, 0.5)
    data_300 = get_processed_data(subject, 0.3)
    data_200 = get_processed_data(subject, 0.2)
    
    if data_500[0] is None or data_300[0] is None or data_200[0] is None:
        print(f"Skipping {subject} due to missing data.")
        return []
        
    X500, y500, trials500 = data_500
    X300, y300, trials300 = data_300
    X200, y200, trials200 = data_200
    
    results = []
    
    # Trial-level randomized split
    unique_trials = np.unique(trials500)
    np.random.seed(42)
    np.random.shuffle(unique_trials)
    n_trials = len(unique_trials)
    n_train = int(0.70 * n_trials)
    n_val = int(0.15 * n_trials)
    
    train_trials = set(unique_trials[:n_train])
    val_trials = set(unique_trials[n_train:n_train+n_val])
    test_trials = set(unique_trials[n_train+n_val:])
    
    # Loop over class ratios
    for ratio in [1, 2]:
        # Balance splits for 500ms
        train_idx_500 = [i for i, tid in enumerate(trials500) if tid in train_trials]
        val_idx_500 = [i for i, tid in enumerate(trials500) if tid in val_trials]
        test_idx_500 = [i for i, tid in enumerate(trials500) if tid in test_trials]
        
        np.random.seed(42)
        train_indices_500 = balance_indices(train_idx_500, y500, ratio)
        val_indices_500 = balance_indices(val_idx_500, y500, ratio)
        test_indices_500 = balance_indices(test_idx_500, y500, ratio)
        
        # Balance splits for 300ms
        train_idx_300 = [i for i, tid in enumerate(trials300) if tid in train_trials]
        val_idx_300 = [i for i, tid in enumerate(trials300) if tid in val_trials]
        test_idx_300 = [i for i, tid in enumerate(trials300) if tid in test_trials]
        
        np.random.seed(42)
        train_indices_300 = balance_indices(train_idx_300, y300, ratio)
        val_indices_300 = balance_indices(val_idx_300, y300, ratio)
        test_indices_300 = balance_indices(test_idx_300, y300, ratio)
        
        # Balance splits for 200ms
        train_idx_200 = [i for i, tid in enumerate(trials200) if tid in train_trials]
        val_idx_200 = [i for i, tid in enumerate(trials200) if tid in val_trials]
        test_idx_200 = [i for i, tid in enumerate(trials200) if tid in test_trials]
        
        np.random.seed(42)
        train_indices_200 = balance_indices(train_idx_200, y200, ratio)
        val_indices_200 = balance_indices(val_idx_200, y200, ratio)
        test_indices_200 = balance_indices(test_idx_200, y200, ratio)
        
        # Normalize and extract features (500ms)
        X_train_raw_500 = X500[train_indices_500]
        X_val_raw_500 = X500[val_indices_500]
        X_test_raw_500 = X500[test_indices_500]
        y_train_500 = y500[train_indices_500]
        y_val_500 = y500[val_indices_500]
        y_test_500 = y500[test_indices_500]
        
        n_ch = X_train_raw_500.shape[1]
        means500 = [np.mean(X_train_raw_500[:, c, :]) for c in range(n_ch)]
        stds500 = [np.std(X_train_raw_500[:, c, :]) for c in range(n_ch)]
        
        def norm_dataset(X, ms, ss):
            Xn = X.copy()
            for c in range(n_ch):
                Xn[:, c, :] = (X[:, c, :] - ms[c]) / (ss[c] + 1e-8)
            return Xn
            
        X_tr_norm_500 = norm_dataset(X_train_raw_500, means500, stds500)
        X_v_norm_500 = norm_dataset(X_val_raw_500, means500, stds500)
        X_te_norm_500 = norm_dataset(X_test_raw_500, means500, stds500)
        
        X_tr_f_500 = np.array([extract_features_from_epoch(ep, use_18=True) for ep in X_tr_norm_500])
        X_v_f_500 = np.array([extract_features_from_epoch(ep, use_18=True) for ep in X_v_norm_500])
        X_te_f_500 = np.array([extract_features_from_epoch(ep, use_18=True) for ep in X_te_norm_500])
        
        # Train 500ms model
        clf500 = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', C=0.5, class_weight='balanced', random_state=42))
        ])
        clf500.fit(np.vstack([X_tr_f_500, X_v_f_500]), np.concatenate([y_train_500, y_val_500]))
        
        # Slices & Features for 300ms
        X_train_raw_300 = X300[train_indices_300]
        X_val_raw_300 = X300[val_indices_300]
        X_test_raw_300 = X300[test_indices_300]
        y_train_300 = y300[train_indices_300]
        y_val_300 = y300[val_indices_300]
        y_test_300 = y300[test_indices_300]
        
        means300 = [np.mean(X_train_raw_300[:, c, :]) for c in range(n_ch)]
        stds300 = [np.std(X_train_raw_300[:, c, :]) for c in range(n_ch)]
        
        X_tr_norm_300 = norm_dataset(X_train_raw_300, means300, stds300)
        X_v_norm_300 = norm_dataset(X_val_raw_300, means300, stds300)
        X_te_norm_300 = norm_dataset(X_test_raw_300, means300, stds300)
        
        X_tr_f_300 = np.array([extract_features_from_epoch(ep, use_18=True) for ep in X_tr_norm_300])
        X_v_f_300 = np.array([extract_features_from_epoch(ep, use_18=True) for ep in X_v_norm_300])
        X_te_f_300 = np.array([extract_features_from_epoch(ep, use_18=True) for ep in X_te_norm_300])
        
        # Train 300ms model
        clf300 = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', C=0.5, class_weight='balanced', random_state=42))
        ])
        clf300.fit(np.vstack([X_tr_f_300, X_v_f_300]), np.concatenate([y_train_300, y_val_300]))
        
        # Slices & Features for 200ms
        X_train_raw_200 = X200[train_indices_200]
        X_val_raw_200 = X200[val_indices_200]
        X_test_raw_200 = X200[test_indices_200]
        y_train_200 = y200[train_indices_200]
        y_val_200 = y200[val_indices_200]
        y_test_200 = y200[test_indices_200]
        
        means200 = [np.mean(X_train_raw_200[:, c, :]) for c in range(n_ch)]
        stds200 = [np.std(X_train_raw_200[:, c, :]) for c in range(n_ch)]
        
        X_tr_norm_200 = norm_dataset(X_train_raw_200, means200, stds200)
        X_v_norm_200 = norm_dataset(X_val_raw_200, means200, stds200)
        X_te_norm_200 = norm_dataset(X_test_raw_200, means200, stds200)
        
        X_tr_f_200 = np.array([extract_features_from_epoch(ep, use_18=True) for ep in X_tr_norm_200])
        X_v_f_200 = np.array([extract_features_from_epoch(ep, use_18=True) for ep in X_v_norm_200])
        X_te_f_200 = np.array([extract_features_from_epoch(ep, use_18=True) for ep in X_te_norm_200])
        
        # Train 200ms model
        clf200 = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', C=0.5, class_weight='balanced', random_state=42))
        ])
        clf200.fit(np.vstack([X_tr_f_200, X_v_f_200]), np.concatenate([y_train_200, y_val_200]))
        
        # Evaluate cross-duration configurations
        m_500_500 = evaluate_model(clf500, X_te_f_500, y_test_500)
        m_500_300 = evaluate_model(clf500, X_te_f_300, y_test_300)
        m_500_200 = evaluate_model(clf500, X_te_f_200, y_test_200)
        m_300_300 = evaluate_model(clf300, X_te_f_300, y_test_300)
        m_200_200 = evaluate_model(clf200, X_te_f_200, y_test_200)
        
        for name, metrics in [
            ("500->500", m_500_500),
            ("500->300", m_500_300),
            ("500->200", m_500_200),
            ("300->300", m_300_300),
            ("200->200", m_200_200),
        ]:
            results.append({
                "Subject": subject,
                "Ratio": f"1:{ratio}",
                "Evaluation": name,
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
    # Evaluate across all 10 subjects
    for subj in config.SUBJECTS:
        res = run_experiment_for_subject(subj)
        all_results.extend(res)
        
    df = pd.DataFrame(all_results)
    csv_path = r"d:\Temple Project\cross_duration_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved cross duration evaluation results to: {csv_path}")
    
    # Print the aggregate summary table grouped by Evaluation and Ratio
    print("\n" + "="*80)
    print("  CROSS DURATION EVALUATION RESULTS (AVERAGE ACROSS SUBJECTS)")
    print("="*80)
    
    summary = df.groupby(["Evaluation", "Ratio"])[["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "FPR"]].mean().reset_index()
    print(summary.to_string(index=False, formatters={
        "Accuracy": "{:.2%}".format,
        "Balanced_Accuracy": "{:.2%}".format,
        "Precision": "{:.2%}".format,
        "Recall": "{:.2%}".format,
        "F1_Score": "{:.2%}".format,
        "ROC_AUC": "{:.2%}".format,
        "FPR": "{:.2%}".format
    }))
