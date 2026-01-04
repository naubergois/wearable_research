
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_figure_11():
    # Configuration
    fs = 64  # Hz
    duration_sec = 16
    t = np.linspace(0, duration_sec, duration_sec * fs)

    # 1. Generate Synthetic BVP Signal (simulating "physiological oscillations")
    # Base heart rate ~ 75 bpm -> 1.25 Hz
    hr_freq = 1.25 
    
    # Create a realistic-looking pulse wave: sum of sines to approximate systolic peak + dicrotic notch
    pulse = (
        1.0 * np.sin(2 * np.pi * hr_freq * t) + 
        0.5 * np.sin(2 * np.pi * 2 * hr_freq * t + 0.5) +  # Harmonic for shape
        0.2 * np.sin(2 * np.pi * 3 * hr_freq * t + 1.0)
    )
    
    # Add amplitude modulation (respiratory sinus arrhythmia-ish)
    modulation = 1.0 + 0.2 * np.sin(2 * np.pi * 0.2 * t) 
    
    # Add some "phasic events" - bursts of higher intensity
    phasic_events = np.zeros_like(t)
    # Event around 4-6s
    phasic_events[int(4*fs):int(6*fs)] = 0.5 * np.hanning(int(2*fs))
    # Event around 10-12s
    phasic_events[int(10*fs):int(12*fs)] = 0.8 * np.hanning(int(2*fs))
    
    bvp_signal = pulse * modulation + phasic_events
    
    # Normalize (as per description "sinal fisiológico bruto normalizado")
    bvp_normalized = (bvp_signal - np.mean(bvp_signal)) / np.std(bvp_signal)

    # 2. Generate Saliency Map (Attribution)
    # Description: "regiões de maior atribuição concentram-se em intervalos temporais específicos"
    # We will attribute high importance to the peaks of the signal, especially during phasic events
    
    # Start with baseline importance
    saliency = 0.1 * np.abs(bvp_normalized)
    
    # Add "heat" to the specific events mentioned above
    # Higher saliency exactly where the BVP amplitude is largest (peaks)
    # and locally concentrated during the "events"
    
    # Envelope of the signal for general regional importance
    envelope = np.abs(bvp_normalized)
    saliency = envelope ** 2  # Squaring enhances peaks
    
    # Smooth slightly to make it look like a heatmap of attribution
    kernel_size = int(0.1 * fs)
    saliency = np.convolve(saliency, np.ones(kernel_size)/kernel_size, mode='same')
    
    # Normalize saliency for visualization (0 to 1)
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency))

    # 3. Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Top Panel: BVP Signal
    axes[0].plot(t, bvp_normalized, color='#2c3e50', linewidth=1.5)
    axes[0].set_ylabel('Amplitude Normalizada')
    axes[0].set_title('Sinal BVP (Normalizado)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Highlight some "phasic events" visually if needed, but the text says points are highlighted.
    # We'll just let the signal speak for itself as "oscilações periódicas" are visible.

    # Bottom Panel: Saliency Heatmap
    # imshow expects 2D, so we expand dimensions
    im = axes[1].imshow(
        saliency[np.newaxis, :], 
        aspect='auto', 
        cmap='RdBu_r', # Warm colors (Red) for high, Cold (Blue) for low
        extent=[t[0], t[-1], 0, 1],
        vmin=0, vmax=1
    )
    
    axes[1].set_ylabel('Atribuição')
    axes[1].set_xlabel('Tempo (s)')
    axes[1].set_title('Mapa de Calor de Atribuição Temporal')
    axes[1].set_yticks([]) # Hide y ticks for heatmap strip
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Importância Relativa')

    plt.suptitle('Figura 11 – Análise Temporal Local (BVP)', fontsize=14)
    
    # Save
    outfile = "figure_11.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {outfile}")

if __name__ == "__main__":
    generate_figure_11()
