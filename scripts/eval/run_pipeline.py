import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from src import config
from src.preprocessing import load_and_preprocess_raw, estimate_baseline_variance, reject_artifact_windows
from src.features import slice_epochs_from_raw
from src.dataset import split_and_scale_dataset
from src.model import train_svm_with_tuning, evaluate_model

def run_subject_pipeline(subject, split_style):
    print(f"\n==================================================")
    print(f"  RUNNING PIPELINE FOR SUBJECT: {subject} ({split_style.upper()} SPLIT)")
    print(f"==================================================")
    
    # Configuration parameters
    percentile = 95
    use_18 = True
    
    # Compute baseline variance thresholds
    print(f"[{subject}] Estimating baseline variance thresholds (percentile={percentile}%)...")
    all_subj_variances = []
    sessions = ["ses-01", "ses-02", "ses-03"]
    
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
        var_thresholds = np.percentile(combined_vars, percentile, axis=0)
        print(f"  [{subject}] Baseline Variance Thresholds ({percentile}th percentile): "
              f"{list(np.round(var_thresholds*1e12, 2))} uV^2")
    else:
        var_thresholds = None
        print(f"  [{subject}] Warning: No baseline data found. Variance thresholding skipped.")
        
    # Extract raw epochs from sessions
    X_subj = []
    y_subj = []
    trial_ids_subj = []
    session_ids_subj = []
    trial_offset = 0
    
    for idx, session in enumerate(sessions, 1):
        try:
            raw = load_and_preprocess_raw(subject, session)
            
            # Slice positive/negative epochs
            epochs, labels, trial_ids = slice_epochs_from_raw(raw)
            if len(epochs) == 0:
                print(f"  [{subject} | {session}] Warning: No epochs found.")
                continue
                
            # Perform artifact rejection
            epochs_array = np.array(epochs)
            clean_indices = reject_artifact_windows(
                epochs_array, var_thresholds=var_thresholds, threshold_uv=config.ARTIFACT_THRESHOLD
            )
            
            clean_epochs = [epochs[i] for i in clean_indices]
            clean_labels = [labels[i] for i in clean_indices]
            clean_trial_ids = [trial_ids[i] for i in clean_indices]
            
            if len(clean_epochs) == 0:
                print(f"  [{subject} | {session}] Warning: No epochs left after artifact rejection.")
                continue
                
            # Accumulate raw epochs
            X_subj.append(np.array(clean_epochs))
            y_subj.append(np.array(clean_labels))
            # Shift trial IDs to be globally unique
            trial_ids_subj.append(np.array(clean_trial_ids) + trial_offset)
            session_ids_subj.append(np.array([idx] * len(clean_epochs)))
            
            trial_offset = max(trial_ids_subj[-1]) if len(clean_trial_ids) > 0 else trial_offset
            
        except Exception as e:
            print(f"  [{subject} | {session}] Error processing session: {e}")
            
    if len(X_subj) == 0:
        print(f"Error: No data available for subject {subject}.")
        return None
        
    # Combine sessions
    X = np.vstack(X_subj)
    y = np.concatenate(y_subj)
    trial_ids = np.concatenate(trial_ids_subj)
    session_ids = np.concatenate(session_ids_subj)
    
    print(f"\n[{subject}] Combined dataset summary:")
    print(f"  Total Epochs: {X.shape[0]}")
    print(f"  Total Channels: {X.shape[1]}")
    print(f"  Total Time Points: {X.shape[2]}")
    print(f"  Total Unique Trial IDs: {len(np.unique(trial_ids))}")
    print(f"  Raw Class Distribution: Pos (Intent)={np.sum(y == 1)}, Neg (Rest)={np.sum(y == 0)}")
    
    results = []
    
    # Loop over class ratios
    for ratio in config.RATIOS:
        print(f"\n--- Evaluation for Class Ratio 1:{ratio} ({split_style.upper()} SPLIT) ---")
        
        # Split dataset, normalize, and extract features
        X_train, y_train, X_val, y_val, X_test, y_test = split_and_scale_dataset(
            X, y, trial_ids, session_ids, ratio_n=ratio, split_style=split_style, use_18=use_18, random_seed=42
        )
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  Skipping ratio 1:{ratio} due to empty splits.")
            continue
            
        # Train SVM
        clf = train_svm_with_tuning(X_train, y_train, X_val, y_val, subject=subject, random_seed=42)
        
        # Evaluate model
        metrics = evaluate_model(clf, X_test, y_test)
        
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
        
    df_results = pd.DataFrame(results)
    return df_results

if __name__ == "__main__":
    split_styles = ["randomized"]

    
    for style in split_styles:
        summary_reports = {}
        for subj in config.SUBJECTS:
            df_res = run_subject_pipeline(subj, split_style=style)
            if df_res is not None:
                summary_reports[subj] = df_res
                
        # Print summary reports
        for subj, df in summary_reports.items():
            print(f"\n==================================================")
            print(f"  SUMMARY REPORT FOR {subj} ({style.upper()} SPLIT)")
            print(f"==================================================")
            print(df.to_string(index=False, formatters={
                "Accuracy": "{:.2%}".format,
                "Balanced_Accuracy": "{:.2%}".format,
                "Precision": "{:.2%}".format,
                "Recall": "{:.2%}".format,
                "F1_Score": "{:.2%}".format,
                "ROC_AUC": "{:.2%}".format,
                "FPR": "{:.2%}".format
            }))
            
            # Save results to CSV
            csv_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results"), f"{subj}_ratio_results_{style}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")
