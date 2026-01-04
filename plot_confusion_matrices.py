
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def main():
    # Load results
    input_file = "training_results.json"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run train_models.py first.")
        return

    with open(input_file, "r") as f:
        data = json.load(f)

    # Prepare metadata output
    metadata_out = {
        "labels": ["No-Stress", "Stress"],
        "matrices": {}
    }

    # Process LOSO models
    models_to_plot = [
        ("loso_mlp", "MLP (Features) - LOSO Validation"),
        ("loso_lstm", "LSTM (Raw) - LOSO Validation")
    ]

    sns.set_style("white")
    
    for key, title in models_to_plot:
        if key not in data:
            continue
            
        cm_data = data[key]["confusion_matrix"]
        tn, fp, fn, tp = cm_data["tn"], cm_data["fp"], cm_data["fn"], cm_data["tp"]
        
        # specific metadata with F1 and AUC
        metadata_out["matrices"][key] = {
            "title": title,
            "values": [[tn, fp], [fn, tp]],
            "metrics": {
                "f1_stress": data[key]["stress"]["f1"],
                "auc": data[key]["auc"]
            }
        }
        
        # Prepare for plotting
        matrix = [[tn, fp], [fn, tp]]
        df_cm = pd.DataFrame(matrix, index=["Real No-Stress", "Real Stress"],
                             columns=["Pred No-Stress", "Pred Stress"])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14})
        plt.title(title)
        
        outfile = f"figure_confusion_matrix_{key.split('_')[1]}.png" # figure_confusion_matrix_mlp.png
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"Generated {outfile}")

    # Save dedicated metadata
    meta_outfile = "metadata_confusion_matrix.json"
    with open(meta_outfile, "w") as f:
        json.dump(metadata_out, f, indent=4)
    print(f"Exported metadata to {meta_outfile}")

if __name__ == "__main__":
    main()
