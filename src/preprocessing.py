import os
import mne
import numpy as np
from src import config

def load_and_preprocess_raw(subject, session):
    """
    Loads BDF, re-references to EXG channels, filters target channels, downsamples,
    and returns selected channels.
    """
    filepath = os.path.join(
        config.DATA_DIR,
        subject,
        session,
        "eeg",
        f"{subject}_{session}_task-innerspeech_eeg.bdf"
    )
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BDF file not found: {filepath}")
        
    print(f"[{subject} | {session}] Scanning raw channel names...")
    raw_info = mne.io.read_raw_bdf(filepath, preload=False, verbose='WARNING')
    all_channels = raw_info.ch_names
    
    keep_channels = config.CHANNELS_TO_USE + config.REF_CHANNELS + ["Status"]
    exclude_channels = [ch for ch in all_channels if ch not in keep_channels]
    
    print(f"[{subject} | {session}] Loading target channels (excluding {len(exclude_channels)} channels)...")
    raw = mne.io.read_raw_bdf(filepath, preload=True, exclude=exclude_channels, verbose='WARNING')
    
    # Re-reference to average of EXG1 and EXG2
    print(f"[{subject} | {session}] Re-referencing to average of EXG1, EXG2...")
    raw.set_eeg_reference(ref_channels=config.REF_CHANNELS, verbose='WARNING')
    
    # Bandpass filter target channels
    print(f"[{subject} | {session}] Bandpass filtering target channels (0.5 - 45 Hz)...")
    raw.filter(l_freq=config.BANDPASS_FREQ[0], h_freq=config.BANDPASS_FREQ[1], 
               picks=config.CHANNELS_TO_USE, fir_design='firwin', verbose='WARNING')
    
    # Notch filter target channels
    print(f"[{subject} | {session}] Notch filtering target channels (50 Hz)...")
    raw.notch_filter(freqs=config.NOTCH_FREQ, picks=config.CHANNELS_TO_USE, fir_design='firwin', verbose='WARNING')
    
    # Downsample target channels
    print(f"[{subject} | {session}] Downsampling to {config.SFREQ_TARGET} Hz...")
    raw.resample(sfreq=config.SFREQ_TARGET, verbose='WARNING')
    
    # Keep only target channels and status trigger
    print(f"[{subject} | {session}] Selecting channels: {keep_channels}")
    raw.pick_channels(config.CHANNELS_TO_USE + ["Status"], verbose='WARNING')
    
    return raw

def estimate_baseline_variance(raw):
    """
    Extracts variances for each channel from baseline intervals (trigger 13 to 14).
    """
    sfreq = raw.info['sfreq']
    window_samples = int(config.WINDOW_DURATION * sfreq) # Window size in samples
    
    # Find events
    events = mne.find_events(raw, stim_channel="Status", verbose='WARNING')
    
    eeg_data = raw.get_data(picks=config.CHANNELS_TO_USE) # shape: (n_channels, n_times)
    
    variances = []
    
    for i in range(len(events)):
        sample_idx = events[i][0]
        code = events[i][2]
        
        # Start of baseline
        if code == 13:
            # End of baseline
            next_sample = eeg_data.shape[1]
            if i + 1 < len(events) and events[i+1][2] == 14:
                next_sample = events[i+1][0]
                
            # Segment into non-overlapping windows
            for start_idx in range(sample_idx, next_sample - window_samples, window_samples):
                epoch = eeg_data[:, start_idx:start_idx + window_samples]
                if epoch.shape[1] == window_samples:
                    # Calculate variance per channel
                    epoch_var = np.var(epoch, axis=1)
                    variances.append(epoch_var)
                    
    return np.array(variances) if len(variances) > 0 else np.empty((0, len(config.CHANNELS_TO_USE)))

def reject_artifact_windows(epochs_data, var_thresholds=None, threshold_uv=config.ARTIFACT_THRESHOLD):
    """
    Rejects epochs exceeding amplitude and variance thresholds.
    """
    threshold_v = threshold_uv * 1e-6
    clean_indices = []
    
    for i in range(epochs_data.shape[0]):
        epoch = epochs_data[i] # shape: (n_channels, n_times)
        
        # Amplitude threshold check
        if np.max(np.abs(epoch)) > threshold_v:
            continue
            
        # Variance threshold check
        if var_thresholds is not None:
            epoch_var = np.var(epoch, axis=1)
            if np.any(epoch_var > var_thresholds):
                continue
                
        clean_indices.append(i)
        
    print(f"Artifact rejection: Kept {len(clean_indices)}/{epochs_data.shape[0]} epochs "
          f"(amplitude threshold: {threshold_uv} uV, variance thresholds: {list(np.round(var_thresholds*1e12, 2)) if var_thresholds is not None else 'None'} uV^2)")
    return clean_indices
