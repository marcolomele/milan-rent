TREE-BASED MODELS OPTIMIZATION PLAN

1. SETUP AND PREPROCESSING
-------------------------
- Import required libraries and set random seeds
- Data split already achieved: train_df and test_df
- Define evaluation metrics:
    * Primary: Squared loss
    * Secondary: MAE, RMSE
- Setup cross-validation strategy: KFold

2. BASELINE MODELS
-----------------
Create simple baseline models with default parameters:
a. Random Forest
b. XGBoost
c. LightGBM
d. CatBoost

3. HYPERPARAMETER OPTIMIZATION
-----------------------------
For each model type, perform systematic optimization:

A. Random Forest:
    * n_estimators: [100, 200, 300, 500]
    * max_depth: [None, 10, 20, 30]
    * min_samples_split: [2, 5, 10]
    * min_samples_leaf: [1, 2, 4]
    * max_features: ['sqrt', 'log2', None]
    * bootstrap: [True, False]
    * class_weight: [None, 'balanced']

B. XGBoost:
    * learning_rate: [0.01, 0.05, 0.1]
    * n_estimators: [100, 200, 300, 500]
    * max_depth: [3, 5, 7, 9]
    * min_child_weight: [1, 3, 5]
    * subsample: [0.8, 0.9, 1.0]
    * colsample_bytree: [0.8, 0.9, 1.0]
    * gamma: [0, 0.1, 0.2]

C. LightGBM:
    * learning_rate: [0.01, 0.05, 0.1]
    * n_estimators: [100, 200, 300, 500]
    * max_depth: [-1, 5, 7, 9]
    * num_leaves: [31, 50, 100]
    * feature_fraction: [0.8, 0.9, 1.0]
    * bagging_fraction: [0.8, 0.9, 1.0]
    * min_child_samples: [20, 30, 50]

D. CatBoost:
    * learning_rate: [0.01, 0.05, 0.1]
    * iterations: [100, 200, 300, 500]
    * depth: [4, 6, 8, 10]
    * l2_leaf_reg: [1, 3, 5, 7]
    * rsm: [0.8, 0.9, 1.0]  # Random subspace method
    * subsample: [0.8, 0.9, 1.0]
    * min_data_in_leaf: [1, 10, 20]

4. OPTIMIZATION METHODOLOGY
--------------------------
For each model:
1. Use RandomizedSearchCV first to identify promising regions
2. Use GridSearchCV around best parameters from RandomizedSearch
3. Implement early stopping where applicable
4. Use cross-validation to ensure robust results

5. ENSEMBLE METHODS
------------------
After finding best parameters for each model:
1. Create a voting/stacking ensemble of best models
2. Try different weighting schemes
3. Compare with individual model performance

6. MODEL EVALUATION AND VALIDATION
--------------------------------
For final model selection:
1. Evaluate on hold-out test set
2. Analyze:
    - Overall performance metrics
    - Performance across different data segments
    - Feature importance stability
    - Training time and inference speed
    - Model complexity vs performance trade-off