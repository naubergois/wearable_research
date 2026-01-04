
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# Configuration
WESAD_ROOT = "wesad/WESAD" # Correct relative path
TARGET_Subject = "S2" # Example subject

def load_subject_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def generate_synthetic_data(duration_sec=60):
    """Generates synthetic data for demonstration purposes."""
    print("Generating synthetic data for visualization...")
    fs_bvp = 64
    fs_eda = 4
    fs_acc = 32
    
    t_bvp = np.linspace(0, duration_sec, duration_sec * fs_bvp)
    t_eda = np.linspace(0, duration_sec, duration_sec * fs_eda)
    t_acc = np.linspace(0, duration_sec, duration_sec * fs_acc)
    
    # Synthetic BVP (sine wave + noise)
    bvp = np.sin(2 * np.pi * 1.2 * t_bvp) + 0.1 * np.random.randn(len(t_bvp))
    
    # Synthetic EDA (slow varying + peaks)
    eda = 1.0 + 0.5 * np.sin(2 * np.pi * 0.05 * t_eda) + 0.2 * np.abs(np.random.randn(len(t_eda)))
    
    # Synthetic TEMP (constant-ish)
    temp = 32.0 + 0.1 * t_eda + 0.05 * np.random.randn(len(t_eda))
    
    # Synthetic ACC (x, y, z)
    acc = np.random.randn(len(t_acc), 3) * 9.8
    
    return {
        "signal": {
            "wrist": {
                "EDA": eda,
                "TEMP": temp,
                "BVP": bvp,
                "ACC": acc
            }
        }
    }

def plot_signals(data):
    sns.set_style("whitegrid")
    
    wrist = data["signal"]["wrist"]
    eda = wrist["EDA"].flatten() if isinstance(wrist["EDA"], np.ndarray) else np.array(wrist["EDA"])
    temp = wrist["TEMP"].flatten() if isinstance(wrist["TEMP"], np.ndarray) else np.array(wrist["TEMP"])
    bvp = wrist["BVP"].flatten() if isinstance(wrist["BVP"], np.ndarray) else np.array(wrist["BVP"])
    acc = wrist["ACC"]
    
    # Plot EDA
    plt.figure(figsize=(12, 4))
    plt.plot(eda, color='purple')
    plt.title('Electrodermal Activity (EDA)')
    plt.ylabel('$\mu$S')
    plt.xlabel('Time (samples)')
    plt.tight_layout()
    plt.savefig('figure_eda_time.png', dpi=300)
    print("Saved figure_eda_time.png")
    
    # Plot TEMP
    plt.figure(figsize=(12, 4))
    plt.plot(temp, color='orange')
    plt.title('Skin Temperature (TEMP)')
    plt.ylabel('°C')
    plt.xlabel('Time (samples)')
    plt.tight_layout()
    plt.savefig('figure_temp_time.png', dpi=300)
    print("Saved figure_temp_time.png")

    # Plot BVP
    plt.figure(figsize=(12, 4))
    plt.plot(bvp, color='red')
    plt.title('Blood Volume Pulse (BVP)')
    plt.ylabel('Magnitude')
    plt.xlabel('Time (samples)')
    plt.tight_layout()
    plt.savefig('figure_bvp_time.png', dpi=300)
    print("Saved figure_bvp_time.png")

    # Plot ACC
    plt.figure(figsize=(12, 4))
    plt.plot(acc[:, 0], label='X', alpha=0.7)
    plt.plot(acc[:, 1], label='Y', alpha=0.7)
    plt.plot(acc[:, 2], label='Z', alpha=0.7)
    plt.title('Accelerometer (ACC)')
    plt.ylabel('m/s^2')
    plt.xlabel('Time (samples)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figure_acc_time.png', dpi=300)
    print("Saved figure_acc_time.png")

def get_real_data():
    """Attempts to load the first available WESAD subject data."""
    # Check current directory and subdirectories for any S*.pkl
    # We prioritize the configured WESAD_ROOT if possible, but the original logic searched "."
    
    search_paths = [WESAD_ROOT, "."]
    
    for path in search_paths:
        if not os.path.exists(path): continue
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pkl") and file.startswith("S"):
                   try:
                       print(f"Found data file: {file}")
                       full_path = os.path.join(root, file)
                       return load_subject_pkl(full_path)
                   except Exception as e:
                       print(f"Error loading {file}: {e}")
    return None

def main():
    data = get_real_data()
    
    if data is None:
        print("No WESAD .pkl files found locally. Using synthetic data.")
        data = generate_synthetic_data()
        
    plot_signals(data)

if __name__ == "__main__":
    main()
