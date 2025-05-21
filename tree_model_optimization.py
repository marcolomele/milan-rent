#%%
# TREE-BASED MODELS OPTIMIZATION SCRIPT

# 1. SETUP AND PREPROCESSING
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from itertools import product
import json

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler # Added for target scaling

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define evaluation metrics
# Primary: Mean Absolute Error (Negative Mean Absolute Error for scikit-learn's scoring)
# Secondary: MSE, RMSE, R2
SCORING = 'neg_mean_absolute_error' # For RandomizedSearchCV/GridSearchCV

def evaluate_model(y_true, y_pred, model_name="Model", target_scaler=None):
    """Calculates and prints regression metrics.
    If target_scaler is provided, it will inverse transform y_true and y_pred 
    before calculating metrics to report them in the original scale.
    """
    y_true_eval = np.array(y_true).copy() # Work with copies
    y_pred_eval = np.array(y_pred).copy()

    if target_scaler is not None:
        # Inverse transform to original scale
        # Reshape if they are 1D arrays
        if y_true_eval.ndim == 1:
            y_true_eval = y_true_eval.reshape(-1, 1)
        if y_pred_eval.ndim == 1:
            y_pred_eval = y_pred_eval.reshape(-1, 1)

        # Inverse MinMax scaling (applied on log1p transformed data)
        y_true_log_unscaled = target_scaler.inverse_transform(y_true_eval)
        y_pred_log_unscaled = target_scaler.inverse_transform(y_pred_eval)

        # Inverse log1p (expm1)
        y_true_eval = np.expm1(y_true_log_unscaled).flatten()
        y_pred_eval = np.expm1(y_pred_log_unscaled).flatten()
        # print(f"DEBUG: Reporting metrics in original scale for {model_name}")
    # else:
        # print(f"DEBUG: Reporting metrics in transformed scale for {model_name}")

    mse = mean_squared_error(y_true_eval, y_pred_eval)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_eval, y_pred_eval)
    r2 = r2_score(y_true_eval, y_pred_eval)
    
    print(f"--- {model_name} Performance (Original Scale if transformer provided) ---")
    print(f"MAE (Primary): {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("----------------------------- ulcers")
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

# Setup cross-validation strategy
N_SPLITS = 5
cv_strategy = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

print("Setup Complete.")

#%%
# Placeholder for Data Loading and Preprocessing
print("Loading and preparing data...")

n_full_train_samples = 4494
n_features = 108

target_scaler = ...
feature_scaler = ...

# Create dummy features and target for the entire original training set
X_full_train_data = np.random.rand(n_full_train_samples, n_features)
y_full_train_data = np.random.rand(n_full_train_samples) * 100 # Example target values

feature_names = [f'feature_{i+1}' for i in range(n_features)]
X_full_train = pd.DataFrame(X_full_train_data, columns=feature_names)
y_full_train = pd.Series(y_full_train_data, name='target')

# Split the (transformed) data
X_train, X_dev, y_train, y_dev = train_test_split(
    X_full_train, 
    y_full_train, # y_train and y_dev are log1p transformed and scaled
    test_size=0.10, 
    random_state=RANDOM_SEED
)

print(f"Original full train data shape: {X_full_train.shape}")
print(f"New X_train shape: {X_train.shape}, new y_train shape: {y_train.shape} (log1p+scaled)")
print(f"X_dev shape: {X_dev.shape}, y_dev shape: {y_dev.shape} (log1p+scaled)")
print("Data loaded, target transformed (log1p then scaled), and split into train/dev sets.")

#%%
# 2. BASELINE MODELS
# Models will be trained on log1p+scaled target.
# evaluate_model will unscale for reporting.

print("Training Baseline Models...")

baseline_models = {
    "RandomForest": RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1),
    "CatBoost": cb.CatBoostRegressor(random_state=RANDOM_SEED, verbose=0)
}

baseline_results = {}

for name, model in baseline_models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train) # y_train is log1p+scaled
    y_pred_dev = model.predict(X_dev) # Predictions are in log1p+scaled domain
    baseline_results[name] = evaluate_model(y_dev, y_pred_dev, 
                                            model_name=f"Baseline {name}", 
                                            target_scaler=target_scaler)

print("\nBaseline Model Training Complete.")
print("Baseline Results (on Dev Set, metrics in original scale):", baseline_results)

#%%
# 3. HYPERPARAMETER OPTIMIZATION
# RandomizedSearchCV will use SCORING ('neg_mean_absolute_error') on the transformed y values.
# The interpretation of RandomizedSearchCV's best_score_ will be in the transformed domain.
# Final evaluation of the tuned model will be in original scale via evaluate_model.

N_ITER_RANDOM_SEARCH = 50
tuned_models = {}
best_params_all_models = {}

# A. Random Forest
print("\nOptimizing Random Forest...")
rf_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5, 0.7, 1.0],
    'bootstrap': [True, False]
}
rf_random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
    param_distributions=rf_param_grid,
    n_iter=N_ITER_RANDOM_SEARCH,
    cv=cv_strategy,
    scoring=SCORING, # Operates on transformed y
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1
)
rf_random_search.fit(X_train, y_train) # y_train is log1p+scaled
tuned_models["RandomForest"] = rf_random_search.best_estimator_
best_params_all_models["RandomForest"] = rf_random_search.best_params_
print(f"Best RandomForest Params: {rf_random_search.best_params_}")
print(f"Best RandomForest Score (CV {SCORING}, on transformed y): {rf_random_search.best_score_:.4f}")


#%%
# B. XGBoost
print("\nOptimizing XGBoost...")
xgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}
xgb_random_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, objective='reg:squarederror'),
    param_distributions=xgb_param_grid,
    n_iter=N_ITER_RANDOM_SEARCH,
    cv=cv_strategy,
    scoring=SCORING, # Operates on transformed y
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1
)
xgb_random_search.fit(X_train, y_train) # y_train is log1p+scaled
tuned_models["XGBoost"] = xgb_random_search.best_estimator_
best_params_all_models["XGBoost"] = xgb_random_search.best_params_
print(f"Best XGBoost Params: {xgb_random_search.best_params_}")
print(f"Best XGBoost Score (CV {SCORING}, on transformed y): {xgb_random_search.best_score_:.4f}")


#%%
# C. CatBoost
print("\nOptimizing CatBoost...")
cat_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 200, 300, 500],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7],
    'rsm': [0.8, 0.9, 1.0],
    'subsample': [0.8, 0.9, 1.0],
    'min_data_in_leaf': [1, 10, 20]
}
cat_random_search = RandomizedSearchCV(
    cb.CatBoostRegressor(random_state=RANDOM_SEED, verbose=0, allow_writing_files=False),
    param_distributions=cat_param_grid,
    n_iter=N_ITER_RANDOM_SEARCH,
    cv=cv_strategy,
    scoring=SCORING, # Operates on transformed y
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1
)
cat_random_search.fit(X_train, y_train) # y_train is log1p+scaled
tuned_models["CatBoost"] = cat_random_search.best_estimator_
best_params_all_models["CatBoost"] = cat_random_search.best_params_
print(f"Best CatBoost Params: {cat_random_search.best_params_}")
print(f"Best CatBoost Score (CV {SCORING}, on transformed y): {cat_random_search.best_score_:.4f}")

print("\nHyperparameter Optimization Complete.")

#%%
# Evaluate Tuned Models on Dev Set
print("\nEvaluating Tuned Models on Dev Set...")
tuned_results_dev = {}
for name, model in tuned_models.items():
    print(f"Evaluating Tuned {name}...")
    y_pred_dev = model.predict(X_dev) # Predictions are in log1p+scaled domain
    tuned_results_dev[name] = evaluate_model(y_dev, y_pred_dev, 
                                             model_name=f"Tuned {name}", 
                                             target_scaler=target_scaler)

print("\nTuned Model Evaluation on Dev Set Complete.")
print("Tuned Model Dev Set Results (metrics in original scale):", tuned_results_dev)

#%%
# 5. ENSEMBLE METHODS
print("\nCreating Ensemble Model...")
estimators = []
for name, model in tuned_models.items():
    if model is not None:
        estimators.append((name.lower(), model))

if estimators:
    voting_regressor = VotingRegressor(estimators=estimators, n_jobs=-1)
    print("Training Voting Regressor...")
    voting_regressor.fit(X_train, y_train) # y_train is log1p+scaled
    print("Evaluating Voting Regressor on Dev Set...")
    y_pred_ensemble_dev = voting_regressor.predict(X_dev) # Predictions are in log1p+scaled domain
    ensemble_results_dev = evaluate_model(y_dev, y_pred_ensemble_dev, 
                                          model_name="Voting Ensemble", 
                                          target_scaler=target_scaler)
    print("Voting Ensemble Dev Set Results (metrics in original scale):", ensemble_results_dev)
else:
    print("No tuned models available to create an ensemble.")

print("\nEnsemble Modeling Complete.")

#%%
# 6. MODEL EVALUATION AND VALIDATION (Final Summary)
print("\n--- FINAL MODEL PERFORMANCE SUMMARY (Dev Set, metrics in original scale) ---")

print("\nBaseline Models (on Dev Set):")
for name, metrics in baseline_results.items():
    print(f"  {name}: MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")

print("\nTuned Individual Models (on Dev Set):")
for name, metrics in tuned_results_dev.items():
    print(f"  {name}: MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")

if 'ensemble_results_dev' in locals() and ensemble_results_dev:
    print("\nEnsemble Model (Voting Regressor on Dev Set):")
    print(f"  Voting Ensemble: MAE={ensemble_results_dev['mae']:.4f}, R2={ensemble_results_dev['r2']:.4f}")

print("\n--- Further Steps ---")
print("1. Analyze feature importances for the best model(s).")
print("2. Perform GridSearchCV for more fine-grained tuning around the best parameters found by RandomizedSearchCV (optimizing for MAE on transformed y).")
print("3. Experiment with different ensemble weighting strategies, potentially based on MAE performance (on transformed y or original scale after unscaling predictions).")
print("4. If you obtain a separate, final TEST set (with labels), evaluate the chosen model on it (remember to unscale predictions for final reporting).")
print("5. Analyze error patterns (e.g., residuals plot) for the best model on the dev set (using unscaled predictions and true values).")
print("----------------------------------------------------")

print("\nScript Finished.")

#%%
# 7. SAVE MODELS AND PARAMETERS
# ----------------------------
print("\nSaving models and parameters...")

# Create models directory if it doesn't exist
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created directory: {models_dir}")

# Save best individual models
for name, model in tuned_models.items():
    if model is not None:
        model_path = os.path.join(models_dir, f'best_{name.lower()}_model.joblib')
        joblib.dump(model, model_path)
        print(f"Saved {name} model to: {model_path}")

# Save ensemble model if it exists
if 'voting_regressor' in locals() and voting_regressor is not None:
    ensemble_path = os.path.join(models_dir, 'best_ensemble_model.joblib')
    joblib.dump(voting_regressor, ensemble_path)
    print(f"Saved Ensemble model to: {ensemble_path}")

# Save best parameters for each model
best_params_path = os.path.join(models_dir, 'best_parameters.json')
pd.Series(best_params_all_models).to_json(best_params_path)
print(f"Saved best parameters to: {best_params_path}")

# Save feature scaler if it exists (assuming it's defined in your data preprocessing)
if 'feature_scaler' in locals() and feature_scaler is not None:
    scaler_path = os.path.join(models_dir, 'feature_scaler.joblib')
    joblib.dump(feature_scaler, scaler_path)
    print(f"Saved feature scaler to: {scaler_path}")

# Save target transformer if it exists (assuming it's defined in your data preprocessing)
if 'target_scaler' in locals() and target_scaler is not None:
    target_scaler_path = os.path.join(models_dir, 'target_scaler.joblib')
    joblib.dump(target_scaler, target_scaler_path)
    print(f"Saved target scaler to: {target_scaler_path}")

print("\nAll models and parameters have been saved successfully.") 

#%%
# 8. GRANULAR GRID SEARCH AROUND BEST PARAMETERS
# --------------------------------------------

models_dir = 'model-runs'
model_filename = 'best_xgboost_model_293.joblib'
model_path = os.path.join(models_dir, model_filename)

if os.path.exists(model_path):
    loaded_xgb_model = joblib.load(model_path)
    print(f"XGBoost model loaded successfully from {model_path}")
else:
    print(f"Error: Model file not found at {model_path}")
    print("Please ensure you have run the cell to save the models.")

best_params = loaded_xgb_model.get_params()

# Define narrow ranges around best values for fine-tuning
granular_params = {
    'max_depth': [best_params['max_depth'] - 1, best_params['max_depth'], best_params['max_depth'] + 1],
    'min_child_weight': [best_params['min_child_weight'] - 2, best_params['min_child_weight'], best_params['min_child_weight'] + 2],
    'subsample': [max(0.8, best_params['subsample'] - 0.05), best_params['subsample'], min(1.0, best_params['subsample'] + 0.05)],
    'colsample_bytree': [max(0.7, best_params['colsample_bytree'] - 0.05), best_params['colsample_bytree'], min(1.0, best_params['colsample_bytree'] + 0.05)],
    'gamma': [max(0, best_params['gamma'] - 0.1), best_params['gamma'], best_params['gamma'] + 0.1]
}

# Function to perform grid search with early stopping
def granular_grid_search(param_grid, base_params, X_train, y_train, X_dev, y_dev, cv_strategy=cv_strategy, early_stopping=50):
    best_score = float('inf')
    best_params = None
    best_model = None
    best_n_rounds = None
    
    # Generate all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, params_to_test in enumerate(param_combinations, 1):
        # Update parameters
        current_params = base_params.copy()
        current_params.update(params_to_test)
        
        # Create and train model
        model = xgb.XGBRegressor(
            **current_params,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            early_stopping_rounds=early_stopping
        )
        
        # Train with cross-validation
        cv_scores = []
        for train_idx, val_idx in cv_strategy.split(X_train):
            X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(
                X_train_cv, y_train_cv,
                eval_set=[(X_val_cv, y_val_cv)],
                verbose=False
            )
            
            # Get validation score
            y_pred_cv = model.predict(X_val_cv)
            cv_scores.append(mean_absolute_error(y_val_cv, y_pred_cv))
        
        # Average CV score
        score = np.mean(cv_scores)
        
        if score < best_score:
            best_score = score
            best_params = params_to_test
            best_model = model
            best_n_rounds = model.best_iteration
            print(f"\nNew best score ({i}/{len(param_combinations)}): {best_score}")
            print("Parameters:", best_params)
            print(f"Best number of rounds: {best_n_rounds}")
    
    return best_model, best_params, best_score, best_n_rounds

# Run granular grid search
print("\nPerforming granular grid search...")
final_model, final_params, final_score, optimal_rounds = granular_grid_search(
    granular_params,
    best_params,
    X_train,
    y_train,
    X_dev,
    y_dev
)

# Print final results
print('\nFinal Results after Granular Search ==========================')
print(f'Best validation score (MAE): {final_score}')
print('Final parameters ---------------------------')
best_params.update(final_params)
for k, v in best_params.items():
    print(k, ':', v)
print(f'Best number of rounds: {optimal_rounds}')

# Save best configuration
config_path = os.path.join(models_dir, 'best_xgboost_config.json')
config = {
    'parameters': best_params,
    'optimal_rounds': optimal_rounds,
    'validation_score': final_score
}
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)
print(f'\nSaved best configuration to: {config_path}')


# Train final model on full training data with best parameters
print('\nTraining final model on full training data...')
ultimate_model = xgb.XGBRegressor(
    **best_params,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
ultimate_model.fit(X_full_train, y_full_train)

# Save the ultimate model
ultimate_model_path = os.path.join(models_dir, 'ultimate_xgboost_model.joblib')
joblib.dump(ultimate_model, ultimate_model_path)
print(f'\nSaved ultimate model to: {ultimate_model_path}')