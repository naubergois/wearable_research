
import json
import numpy as np
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
    # Data for Time Domain Signals (Synthetic)
    # Re-using generation logic from visualize_signals to ensure consistency
    # Note: visualized_signals.generate_synthetic_data returns a nested dict
    
    # We will generate a fresh batch. Since it's random, it won't be *identical* 
    # to the png if we don't fix seed, but for "metadata export" usually representative data is fine
    # or we can fix seed.
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
