import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_figure_11_from_json():
    print("Regenerating Figure 11 from JSON (English Labels)...")
    
    # Path to JSON
    json_path = "resultados/metadata_figure_11.json"
    if not os.path.exists(json_path):
        # Fallback to current dir if not in 'resultados'
        json_path = "metadata_figure_11.json"
        
    print(f"Loading {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
        
    signal = np.array(data["signal"])
    saliency = np.array(data["saliency"])
    
    # Create Time array (assuming 4Hz or inferred from logic? LSTM input is 240 samples @ 4Hz = 60s)
    # The length of signal is 486 in the json?
    # Wait, 486 samples?
    # Train input is 240. Maybe this is a different window?
    # Let's plot by 'Sample Index' if time is unknown.
    t = np.arange(len(signal))
    
    # Plotting
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    # Signal Plot
    ax0 = plt.subplot(gs[0])
    ax0.plot(t, signal, label='BVP (Normalized)', color='blue', linewidth=1.5)
    ax0.set_ylabel('Amplitude (Norm.)', fontsize=12)
    ax0.set_title('BVP Signal & Temporal Importance (Saliency)', fontsize=14)
    ax0.grid(True, linestyle='--', alpha=0.6)
    ax0.legend(loc='upper right')
    ax0.set_xlim([t[0], t[-1]])
    
    # Heatmap
    ax1 = plt.subplot(gs[1])
    # Expand dims for heatmap (1, length)
    saliency_map = saliency.reshape(1, -1)
    
    # Create custom colormap (white -> red)
    cmap = plt.cm.Reds
    
    im = ax1.imshow(saliency_map, aspect='auto', cmap=cmap, vmin=0, vmax=saliency.max())
    ax1.set_xlabel('Time (Samples)', fontsize=12)
    ax1.set_yticks([]) # Hide Y ticks for heatmap stripe
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, orientation='horizontal', fraction=0.5, pad=0.5)
    cbar.set_label('Importance', fontsize=10)
    
    out_img = "resultados/figure_11.png"
    if not os.path.exists("resultados"):
        os.makedirs("resultados")
        
    plt.tight_layout()
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Saved {out_img}")

if __name__ == "__main__":
    plot_figure_11_from_json()
