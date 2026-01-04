
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
import os
import pickle
import json
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Configuration
WESAD_ROOT = "wesad/WESAD"
RAW_SHAPE = (64, 6) 
FEAT_SHAPE = (22,)
N_SAMPLES = 8447
DEMO_MODE = False

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class LSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, 32, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, seq, feature)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        # Take last time step? Or pool? 
        # Typically last time step for classification
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def load_subject_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def generate_synthetic_dataset():
    """Generates synthetic data matching the user's specific dimensions."""
    print(f"Generating synthetic dataset with {N_SAMPLES} samples...")
    
    # 1. Random Features (N, 22)
    y = np.random.choice([0, 1], size=N_SAMPLES, p=[0.7, 0.3]) 
    groups = np.random.randint(2, 18, size=N_SAMPLES) 
    
    X_feat = np.random.randn(N_SAMPLES, 22)
    X_feat[y == 1] += 0.5
    
    # 2. Random Raw Signals (N, 64, 6)
    X_raw = np.random.randn(N_SAMPLES, 64, 6)
    # Add some signal to stress class
    X_raw[y == 1, :, 0] += np.sin(np.linspace(0, 10, 64)) * 0.5
    
    return X_raw, X_feat, y, groups

def build_mlp(input_shape=None):
    # Using Sklearn MLPClassifier
    # Note: Sklearn models are instantiated, not "built" like Keras
    # We return a fresh instance
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.0001, batch_size=64, learning_rate_init=0.001, max_iter=200, random_state=42)

def build_complex_model(input_shape=None):
    # Returning uninitialized class or instance?
    # Better to return instance, but we need to handle "fit" manually
    # We'll return the instance.
    return LSTMModel(input_dim=6)

def get_metrics(y_true, y_pred, y_prob=None):
    """Calculates granular metrics for JSON export including AUC."""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        "accuracy": float(acc),
        "auc": float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None,
        "no_stress": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0])
        },
        "stress": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1])
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        }
    }
    return metrics

def run_loso_validation(X, y, groups, model_builder, model_name="Model"):
    logo = LeaveOneGroupOut()
    
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    
    print(f"\nScanning subjects for {model_name} LOSO...")
    
    split_count = logo.get_n_splits(groups=groups)
    print(f"Total Subjects: {split_count}")

    for i, (train_idx, test_idx) in tqdm(enumerate(logo.split(X, y, groups)), total=split_count, desc=f"{model_name} LOSO"):
        if DEMO_MODE and i >= 3: 
            print("DEMO MODE: Stopping after 3 subjects.")
            break
            
        subject_id = np.unique(groups[test_idx])[0]
        # print(f"Validating on Subject S{subject_id} ({i+1}/{split_count})...")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = model_builder(None)

        if isinstance(model, nn.Module):
            # --- PyTorch Training ---
            model.to(device)
            
            # Prepare Tensors (Keep on CPU until batching usually, but for small data, GPU is fine)
            # data is small enough to fit in GPU mem (350k floats ~ 1.4MB). 
            # So lets move full tensors to device if we want or just batch.
            # DataLoader with TensorDataset on GPU tensors is fast.
            
            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            dataset = TensorDataset(X_train_t, y_train_t)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            epochs = 2 if DEMO_MODE else 15
            
            model.train()
            for ep in range(epochs):
                for xb, yb in dataloader:
                    # xb, yb already on device
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
            
            # Predict
            model.eval()
            with torch.no_grad():
                pred_prob = model(X_test_t).cpu().detach().numpy().flatten()
            
            # Move model back to cpu for saving? Or save state_dict direct.
            model.cpu() 
            
        else:
            # --- Sklearn/Other Training ---
            # Flatten for non-deep models (RF)
            if len(X_train.shape) > 2:
                nsamples, nx, ny = X_train.shape
                X_train = X_train.reshape((nsamples, nx*ny))
                nsamples_test, _, _ = X_test.shape
                X_test = X_test.reshape((nsamples_test, nx*ny))
                
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                pred_prob = model.predict_proba(X_test)[:, 1]
            else:
                pred_prob = model.predict(X_test)

        y_pred = (pred_prob >= 0.5).astype(int)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(pred_prob)
    
    # Save Model
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), f"{model_name.split()[0].lower()}_model.pth")
    else:
        with open(f"{model_name.split()[0].lower()}_model.pkl", "wb") as f:
            pickle.dump(model, f)
            
    print(f"Saved last {model_name} model")
    print(f"{model_name} LOSO Complete.")
    return np.array(y_true_all), np.array(y_pred_all), np.array(y_prob_all)

def run_standard_validation(X, y, model_builder, model_name="Model"):
    """Standard Train/Test Split (Non-LOSO) validation."""
    print(f"\nRunning Standard Validation (Non-LOSO) for {model_name}...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    model = model_builder(None)

    if isinstance(model, nn.Module):
        # PyTorch
        model.to(device)
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        epochs = 2 if DEMO_MODE else 15
        
        model.train()
        for ep in range(epochs):
            for xb, yb in dataloader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred_prob = model(X_test_t).cpu().detach().numpy().flatten()
        model.cpu()
            
    else:
        # Sklearn/RF
        if len(X_train.shape) > 2:
            nsamples, nx, ny = X_train.shape
            X_train = X_train.reshape((nsamples, nx*ny))
            nsamples_test, _, _ = X_test.shape
            X_test = X_test.reshape((nsamples_test, nx*ny))

        model.fit(X_train, y_train)
        
        if hasattr(model, "predict_proba"):
            pred_prob = model.predict_proba(X_test)[:, 1]
        else:
            pred_prob = model.predict(X_test)
        
    y_pred = (pred_prob >= 0.5).astype(int)
    
    print(f"{model_name} Standard Validation Complete.")
    
    return y_test, y_pred, pred_prob

from wesad_data import load_data

def main():
    # --- LOSO Validation ---
    print("\n--- Starting LOSO Validation ---")

    # 1. Processing MLP (Features)
    print("\n[1/2] Processing MLP (Features)...")
    results = {}
    try:
        X_feat, y_feat, groups_feat = load_data(mode="features")
        print(f"Features Loaded: {X_feat.shape}")
        
        y_true, y_pred, y_prob = run_loso_validation(X_feat, y_feat, groups_feat, build_mlp, "MLP (Features)")
        results["loso_mlp"] = get_metrics(y_true, y_pred, y_prob)
        
        # Standard Validation for MLP
        y_true_std, y_pred_std, y_prob_std = run_standard_validation(X_feat, y_feat, build_mlp, "MLP (Features)")
        results["standard_mlp"] = get_metrics(y_true_std, y_pred_std, y_prob_std)
        
        # Free memory
        del X_feat, y_feat, groups_feat
        gc.collect()
        print("Features data cleared from memory.")
        
    except Exception as e:
        print(f"Failed MLP processing: {e}")

    # 2. Processing LSTM (PyTorch)
    print("\n[2/2] Processing LSTM (PyTorch)...")
    try:
        X_raw, y_raw, groups_raw = load_data(mode="raw")
        print(f"Raw Data Loaded: {X_raw.shape}")
        
        y_true, y_pred, y_prob = run_loso_validation(X_raw, y_raw, groups_raw, build_complex_model, "LSTM (PyTorch)")
        results["loso_lstm"] = get_metrics(y_true, y_pred, y_prob)
        
        # Standard Validation for LSTM
        y_true_std, y_pred_std, y_prob_std = run_standard_validation(X_raw, y_raw, build_complex_model, "LSTM (PyTorch)")
        results["standard_lstm"] = get_metrics(y_true_std, y_pred_std, y_prob_std)
        
        del X_raw, y_raw, groups_raw
        gc.collect()
        
    except Exception as e:
        print(f"Failed LSTM processing: {e}")

    # Export
    outfile = "training_results.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults exported to {outfile}")

    # Print summary comparison (Stress AUC & F1)
    print("\n=== Summary (Stress Class) ===")
    print(f"{'Model':<15} | {'Metric':<10} | {'Value':<8}")
    print("-" * 40)
    if 'loso_mlp' in results:
        print(f"{'LOSO MLP':<15} | {'F1-Stress':<10} | {results['loso_mlp']['stress']['f1']:.4f}")
        print(f"{'LOSO MLP':<15} | {'AUC':<10} | {results['loso_mlp']['auc']:.4f}")
    if 'loso_lstm' in results:
        print(f"{'LOSO LSTM':<15} | {'F1-Stress':<10} | {results['loso_lstm']['stress']['f1']:.4f}")
        print(f"{'LOSO LSTM':<15} | {'AUC':<10} | {results['loso_lstm']['auc']:.4f}")

if __name__ == "__main__":
    main()
