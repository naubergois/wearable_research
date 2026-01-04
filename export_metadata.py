
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import visualize_signals
import generate_figure_11

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_figure_10_data():
    # Data from Figure 10 (Feature Importance)
    return {
        "title": "Atribuição média de importância das características (Integrated Gradients)",
        "type": "bar_horizontal",
        "data": {
            "features": [
                'HR', 'TEMP_slope', 'ACC_min', 'BVP_std', 'ACC_std',
                'EDA_slope', 'EDA_mean', 'EDA_std', 'EDA_min', 'EDA_max', 'EDA_peaks'
            ],
            "importance": [
                0.0893, 0.0850, 0.0421, 0.0380, 0.0350,
                0.0200, 0.0180, 0.0150, 0.0100, 0.0080, 0.0050
            ]
        },
        "labels": {
            "x": "Feature Importance",
            "y": "Feature"
        }
    }

def get_time_domain_data():
    # Try to load real data
    data = visualize_signals.get_real_data()
    
    if data:
        print("Using Real WESAD Data for Time Domain Metadata...")
        wrist = data["signal"]["wrist"]
        
        # Slice 60 seconds (approx)
        # Assuming 700Hz is base? No, wrist data has dict of signals with diff frequencies.
        # EDA: 4Hz, TEMP: 4Hz, BVP: 64Hz, ACC: 32Hz
        # We need to slice carefully.
        
        # Let's take a slice from the middle to avoid startup artifacts
        duration_sec = 60
        
        # We'll just take the first N samples corresponding to 60s for each
        # But let's offset by 5 minutes (300s) to be safe?
        offset_sec = 300
        
        eda_sl = slice(offset_sec * 4, (offset_sec + duration_sec) * 4)
        temp_sl = slice(offset_sec * 4, (offset_sec + duration_sec) * 4)
        bvp_sl = slice(offset_sec * 64, (offset_sec + duration_sec) * 64)
        acc_sl = slice(offset_sec * 32, (offset_sec + duration_sec) * 32)
        
        # Safety check on lengths
        if len(wrist["EDA"]) < (offset_sec + duration_sec) * 4:
            offset_sec = 0 # Fallback to start
            eda_sl = slice(0, duration_sec * 4)
            temp_sl = slice(0, duration_sec * 4)
            bvp_sl = slice(0, duration_sec * 64)
            acc_sl = slice(0, duration_sec * 32)
        
        return {
        "title": "Time Domain Signals (Real WESAD - S2)",
        "type": "time_series",
        "sampling_rates": {
            "EDA": 4,
            "TEMP": 4,
            "ACC": 32,
            "BVP": 64
        },
        "signals": {
            "EDA": wrist["EDA"].flatten()[eda_sl],
            "TEMP": wrist["TEMP"].flatten()[temp_sl],
            "BVP": wrist["BVP"].flatten()[bvp_sl],
            "ACC_X": wrist["ACC"][acc_sl, 0],
            "ACC_Y": wrist["ACC"][acc_sl, 1],
            "ACC_Z": wrist["ACC"][acc_sl, 2]
        }
    }
        
    print("Real data not found, using synthetic...")
    np.random.seed(42) 
    data = visualize_signals.generate_synthetic_data(duration_sec=60)
    wrist = data["signal"]["wrist"]
    
    return {
        "title": "Time Domain Signals (Synthetic)",
        "type": "time_series",
        "sampling_rates": {
            "EDA": 4,
            "TEMP": 4,
            "ACC": 32,
            "BVP": 64
        },
        "signals": {
            "EDA": wrist["EDA"].flatten(),
            "TEMP": wrist["TEMP"].flatten(),
            "BVP": wrist["BVP"].flatten(),
            "ACC_X": wrist["ACC"][:, 0],
            "ACC_Y": wrist["ACC"][:, 1],
            "ACC_Z": wrist["ACC"][:, 2]
        }
    }

def get_figure_11_data():
    # Data for Figure 11 (BVP Temporal Saliency)
    # Replicating logic from generate_figure_11.py
    fs = 64
    duration_sec = 16
    t = np.linspace(0, duration_sec, duration_sec * fs)
    
    # Re-implement generation with same parameters
    hr_freq = 1.25 
    pulse = (
        1.0 * np.sin(2 * np.pi * hr_freq * t) + 
        0.5 * np.sin(2 * np.pi * 2 * hr_freq * t + 0.5) +
        0.2 * np.sin(2 * np.pi * 3 * hr_freq * t + 1.0)
    )
    modulation = 1.0 + 0.2 * np.sin(2 * np.pi * 0.2 * t)
    phasic_events = np.zeros_like(t)
    phasic_events[int(4*fs):int(6*fs)] = 0.5 * np.hanning(int(2*fs))
    phasic_events[int(10*fs):int(12*fs)] = 0.8 * np.hanning(int(2*fs))
    
    bvp_signal = pulse * modulation + phasic_events
    bvp_normalized = (bvp_signal - np.mean(bvp_signal)) / np.std(bvp_signal)
    
    envelope = np.abs(bvp_normalized)
    saliency = envelope ** 2
    kernel_size = int(0.1 * fs)
    saliency = np.convolve(saliency, np.ones(kernel_size)/kernel_size, mode='same')
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency))
    
    return {
        "title": "Análise Temporal Local (BVP)",
        "type": "signal_with_heatmap",
        "sampling_rate": fs,
        "duration_sec": duration_sec,
        "data": {
            "time": t,
            "signal_normalized": bvp_normalized,
            "saliency_map": saliency
        },
        "labels": {
            "x": "Tempo (s)",
            "y_signal": "Amplitude Normalizada",
            "y_heatmap": "Atribuição"
        }
    }

def main():
    metadata = {
        "figure_10": get_figure_10_data(),
        "time_domain_signals": get_time_domain_data(),
        "figure_11": get_figure_11_data()
    }
    
    outfile = "figures_metadata.json"
    with open(outfile, "w") as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
    
    print(f"Metadata exported to {outfile}")

if __name__ == "__main__":
    main()
