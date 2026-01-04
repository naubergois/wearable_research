
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import json
from train_models import generate_synthetic_dataset

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_figure_11():
    print("Calculating Real Saliency Map (Gradients)...")
    
    # 1. Load Model
    try:
        model = tf.keras.models.load_model("lstm_model.h5")
    except Exception as e:
        print(f"Error loading model: {e}. Run train_models.py first.")
        return

    # 2. Get Data Sample
    X_raw, _, y, _ = generate_synthetic_dataset()
    
    # Find a stress sample (class 1)
    stress_indices = np.where(y == 1)[0]
    if len(stress_indices) == 0:
        print("No stress samples found.")
        return
    
    sample_idx = stress_indices[0] # Take the first one
    input_sample = X_raw[sample_idx] # (64, 6)
    
    # Preprocess: Expand dim for batch (1, 64, 6)
    input_tensor = tf.convert_to_tensor([input_sample], dtype=tf.float32)
    
    # 3. Calculate Gradients (Saliency)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        preds = model(input_tensor)
        # We want to explain the "Stress" class score
        loss = preds[0][0] # Since sigmoid output is probability of class 1
        
    grads = tape.gradient(loss, input_tensor)
    
    # Saliency = magnitude of gradients
    # We aggregate across channels (last dim) to get temporal importance
    # Shape: (1, 64, 6) -> (64,)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()
    
    # Normalize Saliency
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # Get the BVP signal (index 5 in our synthetic gen: EDA, TEMP, ACCx,y,z, BVP)
    # Wait, in generate_synthetic_dataset: 
    # X_raw = np.random.randn(N_SAMPLES, 64, 6)
    # It doesn't explicitly name cols, but let's assume BVP is the one we modified with sine waves?
    # "X_raw[y == 1, :, 0] += ..." -> Index 0 has the signal added for stress.
    # So let's visualize Index 0 as our "Signal of Interest"
    bvp_signal = input_sample[:, 0]
    
    # Time axis
    t = np.arange(len(bvp_signal))
    
    # 4. Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Signal
    axes[0].plot(t, bvp_signal, color='#2c3e50', linewidth=1.5)
    axes[0].set_ylabel('Amplitude (Norm)')
    axes[0].set_title('Sinal de Entrada (Canal 0)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Saliency
    im = axes[1].imshow(
        saliency[np.newaxis, :], 
        aspect='auto', 
        cmap='RdBu_r', 
        extent=[t[0], t[-1], 0, 1],
        vmin=0, vmax=1
    )
    axes[1].set_ylabel('Atribuição')
    axes[1].set_xlabel('Tempo (amostras)')
    axes[1].set_title('Mapa de Calor de Saliência (Calculado via Gradients)')
    axes[1].set_yticks([])
    
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Importância')
    
    plt.suptitle('Figura 11 – Explicabilidade Real (Gradientes)', fontsize=14)
    
    outfile = "figure_11.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {outfile}")
    
    # Save Metadata
    metadata = {
        "signal": bvp_signal,
        "saliency": saliency,
        "description": "Calculated via tf.GradientTape on trained LSTM model."
    }
    with open("metadata_figure_11.json", "w") as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
    print("Metadata Saved.")

if __name__ == "__main__":
    generate_figure_11()
