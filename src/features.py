import numpy as np
from scipy import signal
import mne
from src import config

def fast_skew(x):
    mu = np.mean(x)
    diff = x - mu
    var = np.mean(diff**2)
    if var == 0:
        return 0.0
    return np.mean(diff**3) / (var**1.5)

def fast_kurtosis(x):
    mu = np.mean(x)
    diff = x - mu
    var = np.mean(diff**2)
    if var == 0:
        return 0.0
    return np.mean(diff**4) / (var**2) - 3.0


def calculate_linear_slope(y):
    """
    Computes the slope of the linear trend in time series y.
    """
    n = len(y)
    x = np.arange(n)
    numerator = n * np.sum(x * y) - np.sum(x) * np.sum(y)
    denominator = n * np.sum(x**2) - (np.sum(x))**2
    if denominator == 0:
        return 0.0
    return numerator / denominator

def calculate_hjorth_parameters(x):
    """
    Computes Hjorth parameters (activity, mobility, complexity).
    """
    diff1 = np.diff(x)
    diff2 = np.diff(diff1)
    
    var_x = np.var(x)
    var_d1 = np.var(diff1)
    var_d2 = np.var(diff2)
    
    if var_x == 0 or var_d1 == 0:
        return 0.0, 0.0, 0.0
        
    mobility = np.sqrt(var_d1 / var_x)
    complexity = np.sqrt(var_d2 / var_d1) / mobility
    return var_x, mobility, complexity

def calculate_spectral_entropy(psd):
    """
    Computes normalized Shannon entropy of the power spectral density (PSD).
    """
    psd_norm = psd / (np.sum(psd) + 1e-12)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return entropy

def extract_features_from_epoch(epoch_data, sfreq=config.SFREQ_TARGET, use_18=True):
    """
    Computes features per channel for a given epoch.
    """
    n_channels, n_times = epoch_data.shape
    features = []
    
    # Welch PSD calculation
    nfft = 256
    freqs, psds = signal.welch(epoch_data, fs=sfreq, nperseg=min(n_times, 125), nfft=nfft, axis=-1)
    
    idx_delta = np.where((freqs >= 0.5) & (freqs < 4.0))[0]
    idx_theta = np.where((freqs >= 4.0) & (freqs <= 8.0))[0]
    idx_alpha = np.where((freqs >= 8.0) & (freqs <= 13.0))[0]
    idx_beta = np.where((freqs >= 13.0) & (freqs <= 30.0))[0]
    idx_gamma = np.where((freqs >= 30.0) & (freqs <= 45.0))[0]
    
    for ch in range(n_channels):
        ch_data = epoch_data[ch]
        
        # Mean
        mean_val = np.mean(ch_data)
        # Variance
        var_val = np.var(ch_data)
        # Linear slope
        slope_val = calculate_linear_slope(ch_data)
        
        ch_psd = psds[ch]
        total_psd_power = np.sum(ch_psd) + 1e-12
        
        power_delta = np.sum(ch_psd[idx_delta])
        power_theta = np.sum(ch_psd[idx_theta])
        power_alpha = np.sum(ch_psd[idx_alpha])
        power_beta = np.sum(ch_psd[idx_beta])
        power_gamma = np.sum(ch_psd[idx_gamma])
        
        # Relative band powers
        rel_delta = power_delta / total_psd_power
        rel_theta = power_theta / total_psd_power
        rel_alpha = power_alpha / total_psd_power
        rel_beta = power_beta / total_psd_power
        rel_gamma = power_gamma / total_psd_power
        
        # Spectral entropy
        spec_entropy = calculate_spectral_entropy(ch_psd)
        
        # Theta, alpha, beta band powers
        theta_power = np.mean(ch_psd[idx_theta]) if len(idx_theta) > 0 else 0.0
        alpha_power = np.mean(ch_psd[idx_alpha]) if len(idx_alpha) > 0 else 0.0
        beta_power = np.mean(ch_psd[idx_beta]) if len(idx_beta) > 0 else 0.0
        
        ch_feats = [
            mean_val, var_val, slope_val, theta_power, alpha_power, beta_power,
            rel_delta, rel_theta, rel_alpha, rel_beta, rel_gamma, spec_entropy
        ]
        
        if use_18:
            # Hjorth parameters
            _, mobility, complexity = calculate_hjorth_parameters(ch_data)
            # Zero crossing rate
            zcr = np.sum(np.diff(np.sign(ch_data)) != 0) / len(ch_data)
            # Higher-order statistical moments and RMS
            skew_val = fast_skew(ch_data)
            kurt_val = fast_kurtosis(ch_data)
            rms = np.sqrt(np.mean(ch_data**2))
            
            ch_feats.extend([mobility, complexity, zcr, skew_val, kurt_val, rms])
            
        features.extend(ch_feats)
        
    return np.array(features)

def slice_epochs_from_raw(raw):
    """
    Slices raw continuous data into positive and negative epochs based on events.
    Excludes covert speech runs.
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
        
        # Track active run type
        if code in [config.RUN_PRONOUNCED, config.RUN_INNER, config.RUN_VISUALIZED]:
            current_run = code
            continue
            
        # Exclude covert runs
        if current_run == config.RUN_INNER:
            continue
            
        # Trial cue onset
        if code in [31, 32, 33, 34]:
            trial_counter += 1
            # Slice negative windows during cue interval
            for start_idx in range(sample_idx, next_sample_idx - window_samples, window_samples):
                epoch = eeg_data[:, start_idx:start_idx + window_samples]
                if epoch.shape[1] == window_samples:
                    epochs_list.append(epoch)
                    labels_list.append(0)
                    trial_ids_list.append(trial_counter)
            continue
            
        # Concentration onset
        if code == 42:
            # Slice negative windows during concentration interval
            for start_idx in range(sample_idx, next_sample_idx - window_samples, window_samples):
                epoch = eeg_data[:, start_idx:start_idx + window_samples]
                if epoch.shape[1] == window_samples:
                    epochs_list.append(epoch)
                    labels_list.append(0)
                    trial_ids_list.append(trial_counter)
            continue
            
        # Action onset
        if code == config.TRIGGER_ACTION_ONSET:
            if current_run == config.RUN_PRONOUNCED:
                # Positive window: pre-action onset
                start_idx = sample_idx - window_samples
                if start_idx >= 0:
                    epoch = eeg_data[:, start_idx:sample_idx]
                    if epoch.shape[1] == window_samples:
                        epochs_list.append(epoch)
                        labels_list.append(1)
                        trial_ids_list.append(trial_counter)
            elif current_run == config.RUN_VISUALIZED:
                # Negative windows during visualization
                for start_idx in range(sample_idx, next_sample_idx - window_samples, window_samples):
                    epoch = eeg_data[:, start_idx:start_idx + window_samples]
                    if epoch.shape[1] == window_samples:
                        epochs_list.append(epoch)
                        labels_list.append(0)
                        trial_ids_list.append(trial_counter)
            continue
            
        # Rest onset
        if code == config.TRIGGER_REST_ONSET:
            # Negative windows during rest (capped at 4s)
            limit_idx = min(next_sample_idx, sample_idx + int(4.0 * sfreq))
            for start_idx in range(sample_idx, limit_idx - window_samples, window_samples):
                epoch = eeg_data[:, start_idx:start_idx + window_samples]
                if epoch.shape[1] == window_samples:
                    epochs_list.append(epoch)
                    labels_list.append(0)
                    trial_ids_list.append(trial_counter)
            continue
            
    return epochs_list, labels_list, trial_ids_list
