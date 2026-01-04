import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from wesad_data import load_data
import json
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_figure_10():
    print("Generating Figure 10 (Real WESAD Permutation Importance)...")
    
    # 1. Load Model (Pickle)
    try:
        with open("mlp_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("Loaded mlp_model.pkl")
    except Exception as e:
        print(f"Error loading model: {e}. Run train_models.py first.")
        return

    # 2. Get Real Data
    try:
        X_feat, y_feat, _ = load_data(mode="features")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    # Standard split to get a validation set for importance calc
    _, X_val, _, y_val = train_test_split(X_feat, y_feat, test_size=0.2, random_state=42)
        
    # 3. Calculate Permutation Importance
    # This replaces Integrated Gradients as a model-agnostic method valid for Sklearn
    print("Calculating permutation importance...")
    r = permutation_importance(model, X_val, y_val,
                               n_repeats=10,
                               random_state=42,
                               n_jobs=-1) # Use parallel
                               
    importance = r.importances_mean
    
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
    
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    df['AbsImportance'] = df['Importance'].abs()
    df = df.sort_values(by='AbsImportance', ascending=True).tail(12) # Top 12
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.barh(df['Feature'], df['AbsImportance'], color=sns.color_palette("magma", len(df)))
    plt.xlabel('Permutation Importance (Mean Decrease in Accuracy)')
    plt.title('Figura 10 – Importância Real (WESAD)')
    plt.tight_layout()
    plt.savefig('figure_10.png', dpi=300)
    print("Figure 10 Saved.")
    
    with open("metadata_figure_10.json", "w") as f:
        json.dump(df.drop(columns=['AbsImportance']).to_dict(orient="list"), f, indent=4, cls=NumpyEncoder)

if __name__ == "__main__":
    generate_figure_10()
