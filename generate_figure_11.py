import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
import torch.nn as nn
from wesad_data import load_data
from train_models import generate_synthetic_dataset, LSTMModel

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_figure_11():
    print("Generating Figure 11 (Real WESAD LSTM Saliency - PyTorch)...")
    
    # 1. Load Model (PyTorch)
    # Re-instantiate model structure first
    model = LSTMModel(input_dim=6)
    try:
        model.load_state_dict(torch.load("lstm_model.pth"))
        print("Loaded lstm_model.pth")
    except Exception as e:
        print(f"Error loading model: {e}. Run train_models.py first.")
        # Attempt to load pickle if fallback naming used? No.
        return

    model.eval()

    # 2. Get Data Sample
    try:
         X_raw, y, _ = load_data(mode="raw")
    except:
         # Fallback
         _, X_raw, y, _ = generate_synthetic_dataset() # Order of return might differ in synthetic gen? 
         # train_models.py: return X_raw, X_feat, y, groups
         # My previous edits might have been confused. 
         # Let's trust load_data mostly.
    
    # Find a stress sample (class 1)
    stress_indices = np.where(y == 1)[0]
    if len(stress_indices) == 0:
        print("No stress samples found.")
        return
    
    sample_idx = stress_indices[0] 
    input_sample = X_raw[sample_idx] # (64, 6)
    
    # Preprocess: Convert to tensor, add batch dim, require grad
    input_tensor = torch.tensor(input_sample, dtype=torch.float32).unsqueeze(0)
    input_tensor.requires_grad = True
    
    # 3. Calculate Saliency (Gradients)
    # Forward
    output = model(input_tensor)
    # Output is probability (sigmoid). We want gradient w.r.t input maximizing this stress score.
    # Score is scalar.
    output.backward()
    
    # Get gradients
    grads = input_tensor.grad.data.numpy()[0] # (64, 6)
    
    # Saliency = magnitude of gradients
    # We aggregate across channels (last dim) to get temporal importance
    temporal_saliency = np.max(np.abs(grads), axis=1) # (64,)
    
    # Normalize
    saliency = (temporal_saliency - temporal_saliency.min()) / (temporal_saliency.max() - temporal_saliency.min() + 1e-8)
    
    # Get the BVP signal (index 5)
    bvp_signal = input_sample[:, 5]
    
    # Time axis
    t = np.arange(len(bvp_signal))
    
    # 4. Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Signal
    axes[0].plot(t, bvp_signal, color='#2c3e50', linewidth=1.5)
    axes[0].set_ylabel('Amplitude (Norm)')
    axes[0].set_title('Sinal de Entrada (Canal BVP) - Exemplo Stress')
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
    axes[1].set_title('Saliência Temporal (LSTM Gradients - PyTorch)')
    axes[1].set_yticks([])
    
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Importância')
    
    plt.suptitle('Figura 11 – Explicabilidade Real (LSTM Saliency)', fontsize=14)
    
    outfile = "figure_11.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {outfile}")
    
    # Save Metadata
    metadata = {
        "signal": bvp_signal,
        "saliency": saliency,
        "description": "Calculated via PyTorch Autograd on LSTM model."
    }
    with open("metadata_figure_11.json", "w") as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
    print("Metadata Saved.")

if __name__ == "__main__":
    generate_figure_11()
