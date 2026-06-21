import numpy as np
from src import config
from src.features import extract_features_from_epoch

def split_and_scale_dataset(X_all, y_all, trial_ids_all, session_ids_all, ratio_n, split_style="randomized", use_18=True, random_seed=42):
    """
    Splits the raw time-series dataset into train, val, and test splits at the trial level.
    Normalizes the raw time series channel-by-channel using training-set statistics,
    and then extracts features from the normalized epochs.
    
    Parameters:
      X_all: 3D numpy array of shape (n_epochs, n_channels, n_times)
      y_all: 1D numpy array of shape (n_epochs,)
      trial_ids_all: 1D numpy array of shape (n_epochs,)
      session_ids_all: 1D numpy array of shape (n_epochs,)
      ratio_n: Integer N representing the 1:N positive-to-negative class ratio
      split_style: 'randomized' or 'session'
      use_18: Boolean whether to extract 18 features (vs 12 features)
      random_seed: Random seed for reproducibility
      
    Returns:
      X_train_features, y_train
      X_val_features, y_val
      X_test_features, y_test
    """
    np.random.seed(random_seed)
    
    if split_style == "randomized":
        # Trial-level split across sessions
        unique_trials = np.unique(trial_ids_all)
        np.random.shuffle(unique_trials)
        
        n_trials = len(unique_trials)
        n_train = int(0.70 * n_trials)
        n_val = int(0.15 * n_trials)
        
        train_trials = set(unique_trials[:n_train])
        val_trials = set(unique_trials[n_train:n_train+n_val])
        test_trials = set(unique_trials[n_train+n_val:])
        
        def get_indices_for_trials(trial_set):
            return [i for i, tid in enumerate(trial_ids_all) if tid in trial_set]
            
        train_idx_all = get_indices_for_trials(train_trials)
        val_idx_all = get_indices_for_trials(val_trials)
        test_idx_all = get_indices_for_trials(test_trials)
        
    elif split_style == "session":
        # Session-based split: Sessions 1 & 2 for train/val, 3 for test
        test_idx_all = [i for i, sid in enumerate(session_ids_all) if sid == 3]
        
        # Train and val splits from sessions 1 and 2
        train_val_idx_all = [i for i, sid in enumerate(session_ids_all) if sid in [1, 2]]
        
        # Split unique train/val trials (85% train, 15% val)
        unique_train_val_trials = np.unique(trial_ids_all[train_val_idx_all])
        np.random.shuffle(unique_train_val_trials)
        
        n_train_trials = int(0.85 * len(unique_train_val_trials))
        train_trials = set(unique_train_val_trials[:n_train_trials])
        val_trials = set(unique_train_val_trials[n_train_trials:])
        
        train_idx_all = [i for i in train_val_idx_all if trial_ids_all[i] in train_trials]
        val_idx_all = [i for i in train_val_idx_all if trial_ids_all[i] in val_trials]
        
    else:
        raise ValueError(f"Unknown split_style: {split_style}")
        
    # Balance classes to 1:N ratio within a split
    def balance_split(indices, split_name):
        if len(indices) == 0:
            return np.empty((0, X_all.shape[1], X_all.shape[2])), np.empty((0,))
            
        split_X = X_all[indices]
        split_y = y_all[indices]
        
        pos_idx = np.where(split_y == 1)[0]
        neg_idx = np.where(split_y == 0)[0]
        
        n_pos = len(pos_idx)
        n_neg_needed = int(n_pos * ratio_n)
        
        if len(neg_idx) < n_neg_needed:
            print(f"  [{split_name}] Warning: Requested {n_neg_needed} negatives, but only {len(neg_idx)} available. Using all.")
            sampled_neg_idx = neg_idx
        else:
            # Sample negatives without replacement
            sampled_neg_idx = np.random.choice(neg_idx, size=n_neg_needed, replace=False)
            
        final_idx = np.concatenate([pos_idx, sampled_neg_idx])
        np.random.shuffle(final_idx) # Shuffle indices
        
        return split_X[final_idx], split_y[final_idx]
        
    # Apply ratio to each split independently
    X_train_raw, y_train = balance_split(train_idx_all, "Train")
    X_val_raw, y_val = balance_split(val_idx_all, "Val")
    X_test_raw, y_test = balance_split(test_idx_all, "Test")
    
    print(f"  Class counts (1:{ratio_n} target ratio):")
    print(f"    Train: {len(y_train)} total (Pos: {np.sum(y_train == 1)}, Neg: {np.sum(y_train == 0)})")
    print(f"    Val:   {len(y_val)} total (Pos: {np.sum(y_val == 1)}, Neg: {np.sum(y_val == 0)})")
    print(f"    Test:  {len(y_test)} total (Pos: {np.sum(y_test == 1)}, Neg: {np.sum(y_test == 0)})")
    
    # Channel-wise normalization using training statistics
    if len(X_train_raw) > 0:
        n_channels = X_train_raw.shape[1]
        means = np.zeros(n_channels)
        stds = np.ones(n_channels)
        
        for ch in range(n_channels):
            # Compute channel statistics across epochs and time points
            ch_data = X_train_raw[:, ch, :]
            means[ch] = np.mean(ch_data)
            stds[ch] = np.std(ch_data)
            
        # Z-score normalize raw channels using training stats
        def normalize_split(X_raw):
            if len(X_raw) == 0:
                return X_raw
            X_norm = X_raw.copy()
            for ch in range(n_channels):
                X_norm[:, ch, :] = (X_raw[:, ch, :] - means[ch]) / (stds[ch] + 1e-8)
            return X_norm
            
        X_train_norm = normalize_split(X_train_raw)
        X_val_norm = normalize_split(X_val_raw)
        X_test_norm = normalize_split(X_test_raw)
    else:
        X_train_norm = X_train_raw
        X_val_norm = X_val_raw
        X_test_norm = X_test_raw
        
    # Feature extraction on normalized time series
    def extract_features(X_norm):
        if len(X_norm) == 0:
            return np.empty((0, len(config.CHANNELS_TO_USE) * (18 if use_18 else 12)))
        features_list = []
        for epoch in X_norm:
            feats = extract_features_from_epoch(epoch, sfreq=config.SFREQ_TARGET, use_18=use_18)
            features_list.append(feats)
        return np.array(features_list)
        
    X_train_features = extract_features(X_train_norm)
    X_val_features = extract_features(X_val_norm)
    X_test_features = extract_features(X_test_norm)
    
    return X_train_features, y_train, X_val_features, y_val, X_test_features, y_test
