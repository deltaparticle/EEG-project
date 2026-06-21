import os

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "ds003626")

# Subjects to process
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-08", "sub-09", "sub-10"]

# Selected electrodes corresponding to: [FCz, Cz, C3, C4]
CHANNELS_TO_USE = ["C23", "A1", "D19", "B22"]

# Re-referencing electrodes (earlobes)
REF_CHANNELS = ["EXG1", "EXG2"]

# Preprocessing Constants
SFREQ_TARGET = 250.0  # Hz
BANDPASS_FREQ = (0.5, 45.0)  # Hz
NOTCH_FREQ = 50.0  # Hz
ARTIFACT_THRESHOLD = 120.0  # uV

# Event Trigger Codes
RUN_PRONOUNCED = 21    # Overt speech run
RUN_VISUALIZED = 23    # Visualized speech run
RUN_INNER = 22         # Covert speech run (excluded)

TRIGGER_ACTION_ONSET = 44  # Action execution cue
TRIGGER_REST_ONSET = 46    # Rest interval

# Classification parameters
RATIOS = [1, 2]  # Positive-to-negative class ratios
WINDOW_DURATION = 0.5  # seconds
