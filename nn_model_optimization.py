#%%
# NEURAL NETWORK (NN) MODELS OPTIMIZATION SCRIPT (PyTorch Version)

# 1. SETUP AND PREPROCESSING
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy # For saving best model
import random # For random search
import json
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler # MinMaxScaler for features and target
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm.auto import tqdm # For progress bars

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Define evaluation metrics
PRIMARY_METRIC = 'mae' # Mean Absolute Error
LOSS_FUNCTION_NAME = 'mean_absolute_error' # For consistency, actual loss is nn.L1Loss

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_nn_model(y_true, y_pred, model_name="NN Model", target_transformer_pt=None):
    """Calculates and prints regression metrics for NN.
    If target_transformer_pt is provided, it will inverse transform y_true and y_pred 
    before calculating metrics to report them in the original scale.
    Assumes y_true and y_pred are in the log1p+scaled domain if transformer is provided.
    """
    y_true_eval_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true).copy()
    y_pred_eval_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.array(y_pred).copy()

    if target_transformer_pt is not None:
        if y_true_eval_np.ndim == 1:
            y_true_eval_np = y_true_eval_np.reshape(-1, 1)
        if y_pred_eval_np.ndim == 1:
            y_pred_eval_np = y_pred_eval_np.reshape(-1, 1)
        
        # Inverse MinMax scaling (applied on log1p transformed data)
        y_true_log_unscaled = target_transformer_pt.inverse_transform(y_true_eval_np)
        y_pred_log_unscaled = target_transformer_pt.inverse_transform(y_pred_eval_np)

        # Inverse log1p (expm1)
        y_true_eval_np = np.expm1(y_true_log_unscaled).flatten()
        y_pred_eval_np = np.expm1(y_pred_log_unscaled).flatten()
        # print(f"DEBUG: NN Reporting metrics in original scale for {model_name}")
    # else:
        # print(f"DEBUG: NN Reporting metrics in transformed scale for {model_name}")

    mae = mean_absolute_error(y_true_eval_np, y_pred_eval_np)
    mse = mean_squared_error(y_true_eval_np, y_pred_eval_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_eval_np, y_pred_eval_np)
    
    print(f"--- {model_name} Performance (Original Scale if transformer provided) ---")
    print(f"MAE (Primary): {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("-----------------------------")
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

print(f"PyTorch Version: {torch.__version__}")
print("Setup Complete.")

#%%
# DATA LOADING AND PREPROCESSING
# ------------------------------
print("Loading and preparing data...")

n_full_train_samples = 4494
n_features = 108

X_full_train_data = np.random.rand(n_full_train_samples, n_features).astype(np.float32)
# Simulate y_train_data being log1p and MinMax scaled
y_original_full_train_data = (np.random.rand(n_full_train_samples) * 100).astype(np.float32) # Original scale
y_log1p_full_train_data = np.log1p(y_original_full_train_data) # Apply log1p

feature_names = [f'feature_{i+1}' for i in range(n_features)]
X_full_train_df = pd.DataFrame(X_full_train_data, columns=feature_names)
# Use y_log1p_full_train_data for the split, then scale y_train and y_dev portions
y_full_train_series_log1p = pd.Series(y_log1p_full_train_data, name='target_log1p')
# Keep original y_dev for final verification if needed, but evaluate_nn_model handles unscaling from transformed y_dev_tensor
y_full_train_series_original = pd.Series(y_original_full_train_data, name='target_original')


X_train_df, X_dev_df, y_train_log1p_series, y_dev_log1p_series, y_train_original_series, y_dev_original_series = train_test_split(
    X_full_train_df,
    y_full_train_series_log1p,
    y_full_train_series_original, # Keep original target for reference
    test_size=0.10,
    random_state=RANDOM_SEED
)

print(f"Original full train data shape: {X_full_train_df.shape}")
print(f"New X_train shape: {X_train_df.shape}, new y_train_log1p shape: {y_train_log1p_series.shape}")
print(f"X_dev (validation) shape: {X_dev_df.shape}, y_dev_log1p (validation) shape: {y_dev_log1p_series.shape}")

feature_scaler = MinMaxScaler()
X_train_scaled_np = feature_scaler.fit_transform(X_train_df)
X_dev_scaled_np = feature_scaler.transform(X_dev_df)

# Scale the log1p transformed target
target_transformer_pt = MinMaxScaler()
y_train_log1p_np = y_train_log1p_series.values.reshape(-1, 1)
y_dev_log1p_np = y_dev_log1p_series.values.reshape(-1, 1)

y_train_transformed_np = target_transformer_pt.fit_transform(y_train_log1p_np)
y_dev_transformed_np = target_transformer_pt.transform(y_dev_log1p_np)

X_train_tensor = torch.tensor(X_train_scaled_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_transformed_np, dtype=torch.float32) # Use transformed target
X_dev_tensor = torch.tensor(X_dev_scaled_np, dtype=torch.float32)
y_dev_tensor = torch.tensor(y_dev_transformed_np, dtype=torch.float32)   # Use transformed target

# Global BATCH_SIZE, can be tuned as well if desired
BATCH_SIZE = 32 
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())

dev_dataset = TensorDataset(X_dev_tensor, y_dev_tensor)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())

print("Data scaled, converted to Tensors, and DataLoaders created.")

#%%
# PyTorch Model Definition
# ------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, config, output_dim=1):
        super(MLP, self).__init__()
        layers_list = [] # Use layers_list instead of layers to avoid name conflict
        current_dim = input_dim
        
        num_hidden_layers = config['num_hidden_layers']
        units_per_layer = config['units_per_layer']
        activations_per_layer = config['activations_per_layer']
        dropout_rates = config['dropout_rates']

        for i in range(num_hidden_layers):
            layers_list.append(nn.Linear(current_dim, units_per_layer[i]))
            if activations_per_layer[i] == 'relu':
                layers_list.append(nn.ReLU())
            elif activations_per_layer[i] == 'tanh':
                layers_list.append(nn.Tanh())
            elif activations_per_layer[i] == 'elu':
                layers_list.append(nn.ELU())
            
            if dropout_rates[i] > 0:
                layers_list.append(nn.Dropout(dropout_rates[i]))
            current_dim = units_per_layer[i]
            
        layers_list.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.network(x)

# Training and Evaluation Loop Functions
# --------------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, epoch_num, num_epochs, trial_num=None):
    model.train()
    total_loss = 0
    total_mae = 0
    
    desc = f"Epoch {epoch_num+1}/{num_epochs} [Train]"
    if trial_num is not None:
        desc = f"Trial {trial_num} - " + desc
    progress_bar = tqdm(dataloader, desc=desc, leave=False)
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        with torch.no_grad():
            mae = nn.L1Loss()(outputs, targets).item()
            total_mae += mae * inputs.size(0)
        progress_bar.set_postfix(loss=loss.item(), mae=mae)
        
    avg_loss = total_loss / len(dataloader.dataset)
    avg_mae = total_mae / len(dataloader.dataset)
    return avg_loss, avg_mae

def validate_epoch(model, dataloader, criterion, device, epoch_num, num_epochs, trial_num=None):
    model.eval()
    total_loss = 0
    total_mae = 0

    desc = f"Epoch {epoch_num+1}/{num_epochs} [Val]"
    if trial_num is not None:
        desc = f"Trial {trial_num} - " + desc    
    progress_bar = tqdm(dataloader, desc=desc, leave=False)
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            mae = nn.L1Loss()(outputs, targets).item()
            total_mae += mae * inputs.size(0)
            progress_bar.set_postfix(loss=loss.item(), mae=mae)
            
    avg_loss = total_loss / len(dataloader.dataset)
    avg_mae = total_mae / len(dataloader.dataset)
    return avg_loss, avg_mae

print("PyTorch model class and train/validation functions defined.")

#%%
# 2. BASELINE NN MODEL (PyTorch)
# ------------------------------
print("\nBuilding and Training Baseline PyTorch NN Model...")

input_dim_baseline = X_train_tensor.shape[1]
baseline_config = {
    'num_hidden_layers': 2,
    'units_per_layer': [64, 32],
    'activations_per_layer': ['relu', 'relu'],
    'dropout_rates': [0.0, 0.0],
    'optimizer_type': 'adam',
    'learning_rate': 0.001
}

baseline_nn_model_pt = MLP(input_dim_baseline, baseline_config).to(device)
optimizer_baseline = optim.Adam(baseline_nn_model_pt.parameters(), lr=baseline_config['learning_rate'])
criterion_baseline = nn.L1Loss()
scheduler_baseline = ReduceLROnPlateau(optimizer_baseline, mode='min', factor=0.2, patience=5, verbose=False)

EPOCHS_BASELINE = 100
PATIENCE_BASELINE = 15
best_val_mae_baseline = float('inf')
patience_counter_baseline = 0
best_baseline_model_state_dict = None

history_baseline_pt = {'loss': [], 'val_loss': [], PRIMARY_METRIC: [], f'val_{PRIMARY_METRIC}': []}

for epoch in range(EPOCHS_BASELINE):
    train_loss, train_mae = train_epoch(baseline_nn_model_pt, train_loader, criterion_baseline, optimizer_baseline, device, epoch, EPOCHS_BASELINE)
    val_loss, val_mae = validate_epoch(baseline_nn_model_pt, dev_loader, criterion_baseline, device, epoch, EPOCHS_BASELINE)
    history_baseline_pt['loss'].append(train_loss); history_baseline_pt[PRIMARY_METRIC].append(train_mae)
    history_baseline_pt['val_loss'].append(val_loss); history_baseline_pt[f'val_{PRIMARY_METRIC}'].append(val_mae)
    print(f"Baseline Epoch {epoch+1}/{EPOCHS_BASELINE} => Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} | Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
    scheduler_baseline.step(val_mae)
    if val_mae < best_val_mae_baseline:
        best_val_mae_baseline = val_mae
        patience_counter_baseline = 0
        best_baseline_model_state_dict = copy.deepcopy(baseline_nn_model_pt.state_dict())
        # torch.save(best_baseline_model_state_dict, 'best_baseline_nn_model_pt.pth') # Save if desired
        print(f"  Baseline Val MAE improved to {val_mae:.4f}.")
    else:
        patience_counter_baseline += 1
        # print(f"  Baseline Val MAE did not improve. Patience: {patience_counter_baseline}/{PATIENCE_BASELINE}")
    if patience_counter_baseline >= PATIENCE_BASELINE: print("Baseline early stopping."); break

print("\nEvaluating Baseline PyTorch NN Model on Dev Set...")
if best_baseline_model_state_dict: baseline_nn_model_pt.load_state_dict(best_baseline_model_state_dict)
baseline_nn_model_pt.eval()
all_preds_baseline = []
all_true_baseline = [] # For collecting y_dev_tensor parts
with torch.no_grad():
    for inputs, targets_batch in dev_loader: # targets_batch will be from y_dev_tensor
        all_preds_baseline.append(baseline_nn_model_pt(inputs.to(device)).cpu())
        all_true_baseline.append(targets_batch.cpu()) # Store the corresponding true values

y_pred_baseline_dev_pt = torch.cat(all_preds_baseline) # Already a tensor
y_true_baseline_dev_pt = torch.cat(all_true_baseline) # Already a tensor

baseline_nn_results_pt = evaluate_nn_model(y_true_baseline_dev_pt, y_pred_baseline_dev_pt, 
                                           model_name="Baseline PyTorch NN", 
                                           target_transformer_pt=target_transformer_pt)

pd.DataFrame(history_baseline_pt).plot(figsize=(8, 5)); plt.title("Baseline PyTorch NN Model Training History (MAE on Transformed Target)")
plt.gca().set_ylim(0, np.max(history_baseline_pt[f'val_{PRIMARY_METRIC}'])*1.2 if history_baseline_pt[f'val_{PRIMARY_METRIC}'] else None); plt.grid(True); plt.show()

#%%
# 3. HYPERPARAMETER OPTIMIZATION (Simulated Randomized Search for PyTorch)
# ----------------------------------------------------------------------
print("\nStarting Hyperparameter Optimization for PyTorch Model...")

param_grid = {
    'num_hidden_layers': [1, 2, 3],
    'units_options': [[32], [64], [128], [256], [64, 32], [128, 64], [256, 128], [128, 64, 32], [256, 128, 64]],
    'activation_options': ['relu', 'tanh', 'elu'],
    'dropout_options': [0.0, 0.1, 0.2, 0.3],
    'learning_rate_options': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'optimizer_options': ['adam', 'rmsprop'] # NAdam can be added if preferred
}

N_RANDOM_TRIALS = 10 # Number of random configurations to try. Increase for better search.
TUNER_EPOCHS = 30    # Max epochs for each trial during tuning. Early stopping is active.
TUNER_PATIENCE = 7   # Patience for early stopping during tuning trials.

all_trial_results = []
best_trial_mae = float('inf')
best_trial_config = None
best_trial_model_state_dict = None

for trial_idx in range(N_RANDOM_TRIALS):
    print(f"\n--- Starting Trial {trial_idx+1}/{N_RANDOM_TRIALS} ---")
    
    # Sample random configuration
    config = {}
    config['num_hidden_layers'] = random.choice(param_grid['num_hidden_layers'])
    
    # Filter units_options based on num_hidden_layers
    possible_units = [u for u in param_grid['units_options'] if len(u) == config['num_hidden_layers']]
    if not possible_units: # Fallback if no exact match (shouldn't happen with good grid design)
        config['units_per_layer'] = random.choice(param_grid['units_options'])[0:config['num_hidden_layers']]
    else:
        config['units_per_layer'] = random.choice(possible_units)
        
    config['activations_per_layer'] = [random.choice(param_grid['activation_options']) for _ in range(config['num_hidden_layers'])]
    config['dropout_rates'] = [random.choice(param_grid['dropout_options']) for _ in range(config['num_hidden_layers'])]
    config['learning_rate'] = random.choice(param_grid['learning_rate_options'])
    config['optimizer_type'] = random.choice(param_grid['optimizer_options'])

    print(f"Trial {trial_idx+1} Config: {config}")

    trial_model = MLP(input_dim_baseline, config).to(device)
    if config['optimizer_type'] == 'adam':
        trial_optimizer = optim.Adam(trial_model.parameters(), lr=config['learning_rate'])
    elif config['optimizer_type'] == 'rmsprop':
        trial_optimizer = optim.RMSprop(trial_model.parameters(), lr=config['learning_rate'])
    # Add other optimizers if in options
    
    trial_criterion = nn.L1Loss()
    trial_scheduler = ReduceLROnPlateau(trial_optimizer, mode='min', factor=0.2, patience=3, verbose=False)

    current_best_trial_epoch_mae = float('inf')
    current_trial_patience_counter = 0
    current_best_trial_epoch_state_dict = None

    for epoch in range(TUNER_EPOCHS):
        train_loss, train_mae = train_epoch(trial_model, train_loader, trial_criterion, trial_optimizer, device, epoch, TUNER_EPOCHS, trial_num=trial_idx+1)
        val_loss, val_mae = validate_epoch(trial_model, dev_loader, trial_criterion, device, epoch, TUNER_EPOCHS, trial_num=trial_idx+1)
        # No extensive history tracking for each trial epoch to save memory, just print
        print(f"  Trial {trial_idx+1} - Epoch {epoch+1}/{TUNER_EPOCHS} => Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")
        trial_scheduler.step(val_mae)

        if val_mae < current_best_trial_epoch_mae:
            current_best_trial_epoch_mae = val_mae
            current_trial_patience_counter = 0
            current_best_trial_epoch_state_dict = copy.deepcopy(trial_model.state_dict())
        else:
            current_trial_patience_counter += 1
        
        if current_trial_patience_counter >= TUNER_PATIENCE:
            print(f"  Trial {trial_idx+1} early stopping at epoch {epoch+1}.")
            break
    
    trial_result = {'config': config, 'val_mae': current_best_trial_epoch_mae, 'state_dict': current_best_trial_epoch_state_dict}
    all_trial_results.append(trial_result)
    print(f"Trial {trial_idx+1} Best Val MAE: {current_best_trial_epoch_mae:.4f}")

    if current_best_trial_epoch_mae < best_trial_mae:
        best_trial_mae = current_best_trial_epoch_mae
        best_trial_config = config
        best_trial_model_state_dict = current_best_trial_epoch_state_dict
        print(f"*** New Best Overall Trial MAE: {best_trial_mae:.4f} ***")

print("\nHyperparameter Optimization (Randomized Search) Complete.")
if best_trial_config:
    print(f"Best Trial Configuration Found:
{best_trial_config}")
    print(f"Best Trial Validation MAE: {best_trial_mae:.4f}")
else:
    print("No successful trials completed or no improvement found.")

#%%
# 4. TRAIN FINAL BEST MODEL (from Randomized Search)
# --------------------------------------------------
if best_trial_config:
    print("\nBuilding and Training Final Best PyTorch NN Model from Search...")
    final_best_model_pt = MLP(input_dim_baseline, best_trial_config).to(device)
    if best_trial_config['optimizer_type'] == 'adam':
        final_optimizer = optim.Adam(final_best_model_pt.parameters(), lr=best_trial_config['learning_rate'])
    elif best_trial_config['optimizer_type'] == 'rmsprop':
        final_optimizer = optim.RMSprop(final_best_model_pt.parameters(), lr=best_trial_config['learning_rate'])

    final_criterion = nn.L1Loss()
    # Use longer patience for final model training
    final_scheduler = ReduceLROnPlateau(final_optimizer, mode='min', factor=0.2, patience=7, verbose=True) 

    FINAL_MODEL_EPOCHS = 150 # Max epochs for final model
    FINAL_MODEL_PATIENCE = 20 # Longer patience for final model
    
    current_best_final_model_val_mae = float('inf')
    final_model_patience_counter = 0
    # Load state from best trial if it makes sense, or train from scratch
    # For simplicity here, we train from scratch using best config. Can also load best_trial_model_state_dict.
    # if best_trial_model_state_dict: final_best_model_pt.load_state_dict(best_trial_model_state_dict)
    
    history_final_best_pt = {'loss': [], 'val_loss': [], PRIMARY_METRIC: [], f'val_{PRIMARY_METRIC}': []}

    for epoch in range(FINAL_MODEL_EPOCHS):
        train_loss, train_mae = train_epoch(final_best_model_pt, train_loader, final_criterion, final_optimizer, device, epoch, FINAL_MODEL_EPOCHS)
        val_loss, val_mae = validate_epoch(final_best_model_pt, dev_loader, final_criterion, device, epoch, FINAL_MODEL_EPOCHS)
        history_final_best_pt['loss'].append(train_loss); history_final_best_pt[PRIMARY_METRIC].append(train_mae)
        history_final_best_pt['val_loss'].append(val_loss); history_final_best_pt[f'val_{PRIMARY_METRIC}'].append(val_mae)
        print(f"Final Model Epoch {epoch+1}/{FINAL_MODEL_EPOCHS} => Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")
        final_scheduler.step(val_mae)
        if val_mae < current_best_final_model_val_mae:
            current_best_final_model_val_mae = val_mae
            final_model_patience_counter = 0
            torch.save(final_best_model_pt.state_dict(), 'final_best_nn_model_pt.pth')
            print(f"  Final Model Val MAE improved to {val_mae:.4f}. Saving model.")
        else:
            final_model_patience_counter += 1
        if final_model_patience_counter >= FINAL_MODEL_PATIENCE: print("Final model early stopping."); break

    print("\nEvaluating Final Best PyTorch NN Model on Dev Set...")
    final_best_model_pt.load_state_dict(torch.load('final_best_nn_model_pt.pth')) # Load best saved state
    final_best_model_pt.eval()
    all_preds_final_best = []
    all_true_final_best = [] # For collecting y_dev_tensor parts
    with torch.no_grad():
        for inputs, targets_batch in dev_loader: # targets_batch will be from y_dev_tensor
            all_preds_final_best.append(final_best_model_pt(inputs.to(device)).cpu())
            all_true_final_best.append(targets_batch.cpu()) # Store the corresponding true values
            
    y_pred_final_best_dev_pt = torch.cat(all_preds_final_best) # Already a tensor
    y_true_final_best_dev_pt = torch.cat(all_true_final_best) # Already a tensor
    
    final_best_nn_results_pt = evaluate_nn_model(y_true_final_best_dev_pt, y_pred_final_best_dev_pt, 
                                                 model_name="Final Best PyTorch NN", 
                                                 target_transformer_pt=target_transformer_pt)

    pd.DataFrame(history_final_best_pt).plot(figsize=(8, 5)); plt.title("Final Best PyTorch NN Model Training History (MAE on Transformed Target)")
    plt.gca().set_ylim(0, np.max(history_final_best_pt[f'val_{PRIMARY_METRIC}'])*1.2 if history_final_best_pt[f'val_{PRIMARY_METRIC}'] else None); plt.grid(True); plt.show()
else:
    print("Skipping final model training as no best trial configuration was found.")
    final_best_nn_results_pt = None # Ensure variable exists

#%%
# 5. MODEL EVALUATION AND VALIDATION (Final Summary on Dev Set)
# -------------------------------------------------------------
print("\n--- FINAL NN MODEL PERFORMANCE SUMMARY (Dev Set) ---")

print("\nBaseline PyTorch NN Model (on Dev Set):")
if 'baseline_nn_results_pt' in locals() and baseline_nn_results_pt:
    print(f"  MAE: {baseline_nn_results_pt['mae']:.4f}, MSE: {baseline_nn_results_pt['mse']:.4f}, R2: {baseline_nn_results_pt['r2']:.4f}")
else:
    print("  Baseline model results not available.")

print("\nFinal Best PyTorch NN Model from Randomized Search (on Dev Set):")
if final_best_nn_results_pt: # Check if it was set
    print(f"  MAE: {final_best_nn_results_pt['mae']:.4f}, MSE: {final_best_nn_results_pt['mse']:.4f}, R2: {final_best_nn_results_pt['r2']:.4f}")
else:
    print("  Final best model results not available (e.g., search yielded no improvements or was skipped).")

print("\n--- Further Steps ---")
print("1. For more robust hyperparameter optimization in PyTorch, use dedicated libraries like Optuna or Ray Tune.")
print("   The randomized search here is a simplified simulation.")
print("2. After finding a promising configuration, consider a more focused 'GridSearch-like' manual tuning around those best parameters.")
print("3. Retrain the absolute best model on the *entire* X_full_train_df and y_full_train_series for a production model.")
print("   (Remember to scale X_full_train_df appropriately). Evaluate on a true unseen TEST set if available.")
print("4. Experiment with batch_size as another hyperparameter to tune.")
print("5. If categorical features are significant, consider using nn.Embedding layers.")
print("6. Analyze error patterns (e.g., residuals plot) for the best model on the dev set.")
print("------------------------------------------------------------------------------------")

print("\nPyTorch NN Model Optimization Script Finished.")

#%%
# 7. SAVE MODELS AND PARAMETERS
# ----------------------------
print("\nSaving models and parameters...")

# Create models directory if it doesn't exist
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created directory: {models_dir}")

# Save baseline model if it exists
if 'baseline_nn_model_pt' in locals() and baseline_nn_model_pt is not None:
    baseline_path = os.path.join(models_dir, 'baseline_nn_model.pt')
    torch.save({
        'model_state_dict': baseline_nn_model_pt.state_dict(),
        'config': baseline_config,
        'performance': baseline_nn_results_pt if 'baseline_nn_results_pt' in locals() else None
    }, baseline_path)
    print(f"Saved baseline model to: {baseline_path}")

# Save final best model if it exists
if 'final_best_model_pt' in locals() and final_best_model_pt is not None:
    final_model_path = os.path.join(models_dir, 'final_best_nn_model.pt')
    torch.save({
        'model_state_dict': final_best_model_pt.state_dict(),
        'config': best_trial_config,
        'performance': final_best_nn_results_pt if 'final_best_nn_results_pt' in locals() else None
    }, final_model_path)
    print(f"Saved final best model to: {final_model_path}")

# Save hyperparameter search results
if 'all_trial_results' in locals() and all_trial_results:
    # Convert trial results to serializable format
    serializable_trials = []
    for trial in all_trial_results:
        serializable_trial = {
            'config': trial['config'],
            'val_mae': float(trial['val_mae'])  # Convert to Python float
        }
        serializable_trials.append(serializable_trial)
    
    trials_path = os.path.join(models_dir, 'hyperparameter_trials.json')
    with open(trials_path, 'w') as f:
        json.dump(serializable_trials, f, indent=2)
    print(f"Saved hyperparameter trials to: {trials_path}")

# Save best trial configuration if it exists
if 'best_trial_config' in locals() and best_trial_config is not None:
    best_config_path = os.path.join(models_dir, 'best_nn_config.json')
    with open(best_config_path, 'w') as f:
        json.dump(best_trial_config, f, indent=2)
    print(f"Saved best configuration to: {best_config_path}")

# Save feature scaler
if 'feature_scaler' in locals() and feature_scaler is not None:
    scaler_path = os.path.join(models_dir, 'feature_scaler.joblib')
    joblib.dump(feature_scaler, scaler_path)
    print(f"Saved feature scaler to: {scaler_path}")

# Save target transformer
if 'target_transformer_pt' in locals() and target_transformer_pt is not None:
    target_transformer_path = os.path.join(models_dir, 'target_transformer.joblib')
    joblib.dump(target_transformer_pt, target_transformer_path)
    print(f"Saved target transformer to: {target_transformer_path}")

print("\nAll models and parameters have been saved successfully.")

# Example code for loading the saved models (commented out)
"""
def load_nn_model(model_path, config_path=None):
    if config_path is None:
        # Load config from model file
        checkpoint = torch.load(model_path)
        config = checkpoint['config']
    else:
        # Load config from separate file
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Initialize model with config
    model = MLP(input_dim=n_features, config=config).to(device)
    # Load state dict
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config, checkpoint.get('performance')
""" 