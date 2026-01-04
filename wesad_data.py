
import os
import pickle
import numpy as np
import subprocess
from glob import glob
from scipy.signal import find_peaks

# --- Configuration ---
WESAD_dataset = "orvile/wesad-wearable-stress-affect-detection-dataset"
WESAD_ROOT = "wesad/WESAD"
BINARY_LABELS = {1:0, 2:1, 3:0}  # 0: Baseline/Amusement, 1: Stress
EDA_FS = 4
BVP_FS = 64
ACC_FS = 32
WINDOW_SEC = 60
STRIDE_SEC = 5

def download_data_if_needed():
    """Checks for dataset and downloads via Kaggle API if missing."""
    if os.path.exists(WESAD_ROOT):
        # Quick check for .pkl files
        if glob(os.path.join(WESAD_ROOT, "**", "*.pkl"), recursive=True):
            return True

    print("WESAD dataset not found/incomplete. Attempting download via Kaggle...")
    
    # Check for kaggle.json in standard locations
    kaggle_locations = ["kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json")]
    has_kaggle_key = any(os.path.exists(loc) for loc in kaggle_locations)
    
    if not has_kaggle_key:
         print("Error: kaggle.json not found in current dir or ~/.kaggle.")
         return False

    # Ensure ~/.kaggle/kaggle.json exists and has correct permissions
    if os.path.exists("kaggle.json") and not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        subprocess.run(["cp", "kaggle.json", os.path.expanduser("~/.kaggle/")])
        subprocess.run(["chmod", "600", os.path.expanduser("~/.kaggle/kaggle.json")])

    try:
        print("Downloading WESAD...")
        subprocess.run(["kaggle", "datasets", "download", "-d", WESAD_dataset, "--unzip", "-p", "wesad"], check=True)
        print("Download complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return False
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Please install: pip install kaggle")
        return False

# --- Feature Extraction Helpers ---

def basic_stats(x):
    if len(x) == 0: return [0.0]*4
    return [float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x))]

def slope_feature(x, fs):
    if len(x) < 2: return 0.0
    return float((x[-1] - x[0]) / (len(x) / fs))

def peak_count(x, distance_samples=5):
    if len(x) == 0: return 0
    peaks, _ = find_peaks(x, distance=distance_samples)
    return int(len(peaks))

def bvp_hr_features(bvp_segment, fs=BVP_FS):
    if len(bvp_segment) < fs * 2: return 0.0, 0.0
    min_distance = int(0.4 * fs)
    peaks, _ = find_peaks(bvp_segment, distance=min_distance)
    if len(peaks) < 2: return 0.0, 0.0
    duration_sec = len(bvp_segment) / fs
    hr_bpm = (len(peaks) / duration_sec) * 60.0
    rr_intervals = np.diff(peaks) / fs
    hrv_sd = float(np.std(rr_intervals)) if len(rr_intervals) > 0 else 0.0
    return float(hr_bpm), hrv_sd

def prepare_signals_and_labels(data):
    """Aligns signals to EDA timeline (4Hz)."""
    wrist = data["signal"]["wrist"]
    eda = wrist["EDA"].flatten()
    temp = wrist["TEMP"].flatten()
    bvp = wrist["BVP"].flatten()
    acc = wrist["ACC"]
    labels = data["label"].flatten()
    
    len_eda = len(eda)
    len_temp = len(temp)
    len_acc = len(acc)
    len_bvp = len(bvp)
    len_labels = len(labels)
    
    idx_eda_to_temp = (np.linspace(0, len_temp - 1, len_eda)).astype(int)
    idx_eda_to_acc  = (np.linspace(0, len_acc  - 1, len_eda)).astype(int)
    idx_eda_to_bvp  = (np.linspace(0, len_bvp  - 1, len_eda)).astype(int)
    idx_eda_to_lbl  = (np.linspace(0, len_labels - 1, len_eda)).astype(int)
    
    temp_aligned   = temp[idx_eda_to_temp]
    acc_aligned    = acc[idx_eda_to_acc]
    bvp_aligned    = bvp[idx_eda_to_bvp] 
    labels_aligned = labels[idx_eda_to_lbl]
    
    return eda, temp_aligned, acc_aligned, bvp_aligned, labels_aligned

def extract_features(eda, temp, acc, bvp_raw, labels):
    """Extracts stats features for MLP."""
    window_samples = WINDOW_SEC * EDA_FS
    stride_samples = STRIDE_SEC * EDA_FS
    
    X_list = []
    y_list = []
    
    len_eda = len(eda)
    len_bvp = len(bvp_raw)
    idx_eda_to_bvp = (np.linspace(0, len_bvp - 1, len_eda)).astype(int)
    
    for start in range(0, len_eda - window_samples, stride_samples):
        end = start + window_samples
        
        lbl_win = labels[start:end]
        valid = lbl_win[np.isin(lbl_win, [1, 2, 3])]
        if len(valid) == 0: continue
        
        bin_labels = [BINARY_LABELS[int(v)] for v in valid]
        label = int(np.mean(bin_labels) >= 0.5)
        
        eda_win = eda[start:end]
        temp_win = temp[start:end]
        acc_win = acc[start:end]
        
        # BVP segment mapping
        bvp_idx_start = idx_eda_to_bvp[start]
        bvp_idx_end = idx_eda_to_bvp[end-1] if end-1 < len(idx_eda_to_bvp) else idx_eda_to_bvp[-1]
        bvp_seg = bvp_raw[bvp_idx_start:bvp_idx_end]
        
        # --- Feature Calculation ---
        eda_f = basic_stats(eda_win) + [slope_feature(eda_win, EDA_FS), peak_count(eda_win, 2)]
        temp_f = basic_stats(temp_win) + [slope_feature(temp_win, EDA_FS)]
        acc_mag = np.linalg.norm(acc_win, axis=1)
        acc_f = basic_stats(acc_mag) + [peak_count(acc_mag, 2)]
        bvp_f = basic_stats(bvp_seg)
        hr, hrv = bvp_hr_features(bvp_seg, BVP_FS)
        
        feats = eda_f + temp_f + acc_f + bvp_f + [hr, hrv]
        
        X_list.append(feats)
        y_list.append(label)
        
    return np.array(X_list), np.array(y_list)

def extract_raw_windows(eda, temp, acc, bvp_raw, labels):
    """Extracts Raw Windows for LSTM (resampled to match feature window logic)."""
    # For LSTM, we might want to feed the raw signals. 
    # To keep it simple and compatible with the MLP pipeline windowing:
    # We will resample everything to a common frequency (e.g., 4Hz) or just stack them?
    # WESAD paper often uses raw signals. 
    # Let's resample all to 4Hz for simplicity in this demo (6 channels: EDA, TEMP, ACCx,y,z, BVP)
    
    window_samples = WINDOW_SEC * EDA_FS
    stride_samples = STRIDE_SEC * EDA_FS
    
    X_raw_list = []
    y_list = []
    
    len_eda = len(eda)
    
    # We assume 'acc' is already aligned to eda in 'acc_aligned' from prepare_signals_and_labels
    # But bvp is raw. We need to resample/align BVP to 4Hz as well for the 6-channel stack.
    len_bvp = len(bvp_raw)
    idx_bvp_to_eda = (np.linspace(0, len_eda - 1, len_bvp)).astype(int) # This is inverse...
    
    # Easier: Just use the 'bvp_aligned' returned by prepare_signals_and_labels which is mapped to EDA timeline (4Hz samples)
    # But wait, prepare_signals_and_labels returns bvp_aligned which picks samples. This is simple downsampling.
    # It satisfies the shape requirement.
    
    # Wait, prepare_signals_and_labels returns (eda, temp_aligned, acc_aligned, bvp_aligned, labels_aligned)
    # Oh wait, the fn signature returns 'bvp' which is the ALIGNED one (mapped to EDA indices).
    # Let's verify prepare_signals_and_labels in THIS file.
    # It returns: eda, temp_aligned, acc_aligned, bvp (THIS IS BVP_ALIGNED), labels_aligned
    pass 

def load_data(mode="features"):
    """
    Main entry point.
    mode: 'features' (MLP) or 'raw' (LSTM)
    Returns: X, y, groups (subject_ids)
    """
    if not download_data_if_needed():
        raise RuntimeError("Could not download WESAD dataset.")

    files = sorted(glob(os.path.join(WESAD_ROOT, "**", "S*.pkl"), recursive=True))
    
    X_all = []
    y_all = []
    groups_all = []
    
    for f in files:
        subject_id = int(os.path.basename(f)[1:-4]) # S2 -> 2
        print(f"Processing Subject S{subject_id}...")
        
        try:
            with open(f, "rb") as pkl:
                data = pickle.load(pkl, encoding="latin1")
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
            
        eda, temp, acc, bvp_aligned, labels = prepare_signals_and_labels(data)
        
        if mode == "features":
            # Pass RAW bvp (from data object) to extract_features because it calculates HR/HRV from 64Hz
            bvp_raw = data["signal"]["wrist"]["BVP"].flatten() 
            X, y = extract_features(eda, temp, acc, bvp_raw, labels)
        else:
            # Raw mode for LSTM: stack channels [EDA, TEMP, ACCx, ACCy, ACCz, BVP]
            # All aligned to 4Hz timeline
            # acc is [N, 3]
            # stack: eda(N,1), temp(N,1), acc(N,3), bvp_aligned(N,1) -> (N, 6)
            
            # Simple windowing on the stacked array
            stack = np.column_stack([eda, temp, acc, bvp_aligned])
            
            window_samples = WINDOW_SEC * EDA_FS
            stride_samples = STRIDE_SEC * EDA_FS
            
            X_wins = []
            y_wins = []
            
            for start in range(0, len(stack) - window_samples, stride_samples):
                end = start + window_samples
                lbl_win = labels[start:end]
                valid = lbl_win[np.isin(lbl_win, [1, 2, 3])]
                if len(valid) == 0: continue
                
                bin_labels = [BINARY_LABELS[int(v)] for v in valid]
                label = int(np.mean(bin_labels) >= 0.5)
                
                win_data = stack[start:end]
                X_wins.append(win_data)
                y_wins.append(label)
            
            X = np.array(X_wins)
            y = np.array(y_wins)
            if mode == "raw":
                print(f"Subject {subject_id}: X raw shape {X.shape}")

        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
            groups_all.append(np.full(len(y), subject_id))
            
    if not X_all:
        raise RuntimeError("No valid data extracted from any subject.")
        
    return np.vstack(X_all), np.concatenate(y_all), np.concatenate(groups_all)
