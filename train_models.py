
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.utils import class_weight
import os
import pickle
import pandas as pd
import json

# Configuration
WESAD_ROOT = "/content/wesad/WESAD"
RAW_SHAPE = (64, 6) # 64 timesteps, 6 channels (EDA, TEMP, ACC_x,y,z, BVP)
FEAT_SHAPE = (22,)  # 22 statistical features
N_SAMPLES = 8447    # Exact number from user request
DEMO_MODE = True    # Set to False for full LOSO validation takes ~1 hour

def load_subject_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def generate_synthetic_dataset():
    """Generates synthetic data matching the user's specific dimensions."""
    print(f"Generating synthetic dataset with {N_SAMPLES} samples...")
    
    # 1. Random Features (N, 22)
    y = np.random.choice([0, 1], size=N_SAMPLES, p=[0.7, 0.3]) # ~30% stress
    groups = np.random.randint(2, 18, size=N_SAMPLES) # Subjects S2..S17
    
    X_feat = np.random.randn(N_SAMPLES, 22)
    # Shift means for stress class to make it learnable
    X_feat[y == 1] += 0.5
    
    # 2. Random Raw Signals (N, 64, 6)
    X_raw = np.random.randn(N_SAMPLES, 64, 6)
    # Add some frequency component to stress class to make it learnable by LSTM
    X_raw[y == 1, :, 0] += np.sin(np.linspace(0, 10, 64)) * 0.5
    
    return X_raw, X_feat, y, groups

def build_mlp(input_shape):
    i = layers.Input(shape=input_shape)
    x = layers.Dense(64, activation="relu")(i)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    o = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(i, o)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=["accuracy"]
    )
    return model

def build_lstm(input_shape):
    i = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(i)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    o = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(i, o)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=["accuracy"]
    )
    return model

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

    for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        if DEMO_MODE and i >= 3: 
            print("DEMO MODE: Stopping after 3 subjects.")
            break
            
        subject_id = np.unique(groups[test_idx])[0]
        print(f"Validating on Subject S{subject_id} ({i+1}/{split_count})...")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        cw = dict(enumerate(weights))
        
        model = model_builder(X.shape[1:])
        
        epochs = 2 if DEMO_MODE else 30
        model.fit(
            X_train, y_train,
            epochs=epochs, 
            batch_size=64,
            verbose=0,
            class_weight=cw
        )
        
        pred_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (pred_prob >= 0.5).astype(int)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(pred_prob)
        
        tf.keras.backend.clear_session()
        
    print(f"{model_name} LOSO Complete.")
    return np.array(y_true_all), np.array(y_pred_all), np.array(y_prob_all)

def run_standard_validation(X, y, model_builder, model_name="Model"):
    """Standard Train/Test Split (Non-LOSO) validation."""
    print(f"\nRunning Standard Validation (Non-LOSO) for {model_name}...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    cw = dict(enumerate(weights))

    model = model_builder(X.shape[1:])
    epochs = 2 if DEMO_MODE else 30
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=64,
        verbose=1 if DEMO_MODE else 0,
        class_weight=cw
    )
    
    pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (pred_prob >= 0.5).astype(int)
    
    print(f"{model_name} Standard Validation Complete.")
    
    tf.keras.backend.clear_session()
    return y_test, y_pred, pred_prob

def main():
    X_raw, X_feat, y, groups = generate_synthetic_dataset()
    
    results = {}

    # --- LOSO Validation ---
    print("\n--- Starting LOSO Validation ---")
    y_true, y_pred, y_prob = run_loso_validation(X_feat, y, groups, build_mlp, "MLP (Features)")
    results["loso_mlp"] = get_metrics(y_true, y_pred, y_prob)
    
    y_true, y_pred, y_prob = run_loso_validation(X_raw, y, groups, build_lstm, "LSTM (Raw)")
    results["loso_lstm"] = get_metrics(y_true, y_pred, y_prob)
    
    # --- Standard Validation (Non-LOSO) ---
    print("\n--- Starting Standard Validation (Non-LOSO) ---")
    y_true, y_pred, y_prob = run_standard_validation(X_feat, y, build_mlp, "MLP (Features)")
    results["standard_mlp"] = get_metrics(y_true, y_pred, y_prob)
    
    y_true, y_pred, y_prob = run_standard_validation(X_raw, y, build_lstm, "LSTM (Raw)")
    results["standard_lstm"] = get_metrics(y_true, y_pred, y_prob)

    # Export
    outfile = "training_results.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults exported to {outfile}")

    # Print summary comparison (Stress AUC & F1)
    print("\n=== Summary (Stress Class) ===")
    print(f"{'Model':<15} | {'Metric':<10} | {'Value':<8}")
    print("-" * 40)
    print(f"{'LOSO MLP':<15} | {'F1-Stress':<10} | {results['loso_mlp']['stress']['f1']:.4f}")
    print(f"{'LOSO MLP':<15} | {'AUC':<10} | {results['loso_mlp']['auc']:.4f}")
    print(f"{'Standard MLP':<15} | {'F1-Stress':<10} | {results['standard_mlp']['stress']['f1']:.4f}")
    print(f"{'Standard MLP':<15} | {'AUC':<10} | {results['standard_mlp']['auc']:.4f}")
    print("-" * 40)
    print(f"{'LOSO LSTM':<15} | {'F1-Stress':<10} | {results['loso_lstm']['stress']['f1']:.4f}")
    print(f"{'LOSO LSTM':<15} | {'AUC':<10} | {results['loso_lstm']['auc']:.4f}")
    print(f"{'Standard LSTM':<15} | {'F1-Stress':<10} | {results['standard_lstm']['stress']['f1']:.4f}")
    print(f"{'Standard LSTM':<15} | {'AUC':<10} | {results['standard_lstm']['auc']:.4f}")

if __name__ == "__main__":
    main()
