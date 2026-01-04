
import json
import os
import numpy as np

# Helper to handle numpy types if any leak through, though loading from JSON they should be standard types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    print("Consolidating XAI Metadata...")
    
    file_fig10 = "metadata_figure_10.json"
    file_fig11 = "metadata_figure_11.json"
    
    xai_data = {
        "feature_importance": None,
        "temporal_saliency": None
    }
    
    # Load Figure 10 (Feature Importance)
    if os.path.exists(file_fig10):
        with open(file_fig10, "r") as f:
            xai_data["feature_importance"] = json.load(f)
        print(f"Loaded {file_fig10}")
    else:
        print(f"Warning: {file_fig10} not found.")

    # Load Figure 11 (Temporal Saliency)
    if os.path.exists(file_fig11):
        with open(file_fig11, "r") as f:
            xai_data["temporal_saliency"] = json.load(f)
        print(f"Loaded {file_fig11}")
    else:
        print(f"Warning: {file_fig11} not found.")
        
    # Validation
    if xai_data["feature_importance"] is None and xai_data["temporal_saliency"] is None:
        print("Error: No XAI metadata found.")
        return

    # Export
    outfile = "metadata_xai.json"
    with open(outfile, "w") as f:
        json.dump(xai_data, f, indent=4, cls=NumpyEncoder)
    
    print(f"Successfully created {outfile}")

if __name__ == "__main__":
    main()
