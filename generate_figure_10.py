
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from wesad_data import load_data
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

@tf.function
def grad_tabular(model, inputs):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        pred = model(inputs)
    return tape.gradient(pred, inputs)

def integrated_gradients(model, baseline, sample, steps=50):
    alphas = tf.linspace(0.0, 1.0, steps)
    grads = []
    for a in alphas:
        x = baseline + a * (sample - baseline)
        g = grad_tabular(model, x)
        grads.append(g)
    avg = tf.reduce_mean(tf.stack(grads), axis=0)
    return (sample - baseline) * avg

def generate_figure_10():
    print("Generating Figure 10 (Real WESAD Integrated Gradients)...")
    
    # 1. Load Model
    try:
        model = tf.keras.models.load_model("mlp_model.h5")
        print("Loaded mlp_model.h5")
    except Exception as e:
        print(f"Error loading model: {e}. Run train_models.py first.")
        return

    # 2. Get Real Data
    try:
        # Load just features
        X_feat, y_feat, _ = load_data(mode="features")
        # Use simple standard scaler since model was likely trained on scaled or unscaled?
        # Note: train_models.py didn't scale effectively (it used class weight but data was just loaded).
        # Actually wesad_data.py extract_features produces raw features.
        # MLP in train_models.py learns on whatever is passed.
        # So we should pass the same distribution.
        
        # We need a TRUE POSITIVE for Stress (class 1)
        # Random sample
        # Let's verify prediction first
        preds_prob = model.predict(X_feat, verbose=0).flatten()
        preds = (preds_prob >= 0.5).astype(int)
        
        # Find True Positives
        tp_indices = np.where((y_feat == 1) & (preds == 1))[0]
        
        if len(tp_indices) == 0:
            print("No True Positive Stress samples found for explanation.")
            # Fallback to just Ground Truth Stress
            tp_indices = np.where(y_feat == 1)[0]
            if len(tp_indices) == 0:
                 print("No Stress samples at all found.")
                 return
        
        # Pick one good sample (e.g. high probability)
        best_idx = tp_indices[np.argmax(preds_prob[tp_indices])]
        sample = X_feat[best_idx:best_idx+1]
        
        # Baseline: Zeros? Or mean? Zeros is standard for IG on signal magnitude/counts features
        baseline = np.zeros_like(sample)
        
        # 3. Calculate IG
        attr = integrated_gradients(
            model, 
            tf.cast(baseline, tf.float32), 
            tf.cast(sample, tf.float32)
        ).numpy()[0]
        
        # 4. Feature Names (Standard WESAD extracted order)
        # eda (4 stats + slope + peaks) = 6
        # temp (4 stats + slope) = 5
        # acc (4 stats + peaks) = 5
        # bvp (4 stats) + hr + hrv = 6
        # Total 22
        feature_names = [
            'EDA_mean', 'EDA_std', 'EDA_min', 'EDA_max', 'EDA_slope', 'EDA_peaks',
            'TEMP_mean', 'TEMP_std', 'TEMP_min', 'TEMP_max', 'TEMP_slope',
            'ACC_mean', 'ACC_std', 'ACC_min', 'ACC_max', 'ACC_peaks',
            'BVP_mean', 'BVP_std', 'BVP_min', 'BVP_max', 'HR', 'HRV'
        ]
        
        df = pd.DataFrame({'Feature': feature_names, 'Importance': attr})
        df['AbsImportance'] = df['Importance'].abs()
        df = df.sort_values(by='AbsImportance', ascending=True).tail(12) # Top 12
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        plt.barh(df['Feature'], df['AbsImportance'], color=sns.color_palette("magma", len(df)))
        plt.xlabel('Integrated Gradients Importance')
        plt.title('Figura 10 – Importância Real (WESAD)')
        plt.tight_layout()
        plt.savefig('figure_10.png', dpi=300)
        print("Figure 10 Saved.")
        
        with open("metadata_figure_10.json", "w") as f:
            json.dump(df.drop(columns=['AbsImportance']).to_dict(orient="list"), f, indent=4, cls=NumpyEncoder)
            
    except Exception as e:
        print(f"Error generating Figure 10: {e}")

if __name__ == "__main__":
    generate_figure_10()
