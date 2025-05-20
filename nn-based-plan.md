NEURAL NETWORK (NN) MODELS OPTIMIZATION PLAN

1. SETUP AND PREPROCESSING
-------------------------
- Import required libraries (TensorFlow/Keras or PyTorch) and set random seeds for reproducibility.
- Data Scaling: features standardised with minmax.
- Data splitting: data available in train_df and test_df (labels not available)
- Evaluation Metrics: Mean Absolute Error (MAE). Secondary: Mean Squared Error (MSE), RMSE, R2 Score.
- Define Callbacks for Training: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`

2. BASELINE NN MODEL
--------------------
- Define a simple, relatively shallow Multi-Layer Perceptron (MLP) architecture as a starting point:
    * Input layer: `input_shape` corresponding to the number of features.
    * 1-2 hidden layers (e.g., 64 units, then 32 units) with ReLU activation.
    * Output layer: 1 unit with a linear activation function (for regression).
    * Optimizer: Adam with a default learning rate (e.g., 0.001).
    * Loss function: 'mean_squared_error' or 'mean_absolute_error'.
- Train this baseline model on `X_train`, `y_train` and evaluate on `X_dev`, `y_dev`. This establishes a performance benchmark.

3. HYPERPARAMETER AND ARCHITECTURE OPTIMIZATION
---------------------------------------------
Systematically explore different aspects of the NN. Use an automated hyperparameter tuning library if possible (e.g., Keras Tuner, Optuna, Ray Tune) or perform manual/grid search.

**A. Architecture Exploration:**
    *   **Number of Hidden Layers:** [1, 2, 3, 4, 5]
    *   **Number of Neurons per Layer:**
        *   Schemes: Constant (e.g., all layers 64 units), decreasing (e.g., 128 -> 64 -> 32), increasing then decreasing (e.g., 64 -> 128 -> 64).
        *   Ranges: [16, 32, 64, 128, 256, 512] units per layer. Consider powers of 2.
    *   **Activation Functions for Hidden Layers:** ['relu', 'elu', 'tanh', 'swish']
    *   **Input Layer Activation:** Typically not used, data is passed directly.
    *   **Output Layer Activation:** 'linear' for regression.

**B. Regularization Techniques (to prevent overfitting):**
    *   **Dropout:**
        *   Placement: After hidden layers.
        *   Dropout Rate: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    *   **L1/L2 Regularization (Kernel Regularizers):**
        *   Apply to weights of dense layers.
        *   Regularization Strength (lambda): [1e-5, 1e-4, 1e-3, 1e-2]
    *   **Batch Normalization:**
        *   Placement: Before or after activation functions in hidden layers. Explore both.

**C. Optimizer and Learning Rate:**
    *   **Optimizers:** ['adam', 'rmsprop', 'sgd' (with momentum), 'nadam']
    *   **Learning Rate:**
        *   Fixed: [0.01, 0.005, 0.001, 0.0005, 0.0001]
        *   Learning Rate Schedules (if not using `ReduceLROnPlateau` callback explicitly): e.g., exponential decay.

4. OPTIMIZATION METHODOLOGY
--------------------------
- **Iterative Approach:** Start with simpler architectures and gradually increase complexity.
- **One Change at a Time (Initially):** When manually tuning, change one hyperparameter at a time to understand its impact.
- **Automated Hyperparameter Tuning:**
    *   Define a search space for the hyperparameters listed above.
    *   Use a tuner (e.g., `RandomSearch`, `Hyperband`, or Bayesian optimization from Keras Tuner/Optuna).
    *   Train multiple trials, evaluating each on the validation set (`X_dev`, `y_dev`).
- **Cross-Validation (Optional but Recommended for Robustness):**
    *   While NNs are often trained with a single validation split due to computational cost, K-Fold CV can provide more robust hyperparameter estimates if resources allow. Each fold would require retraining the NN.
    *   If not using full CV, ensure the dev set is representative.
- **Focus on Validation Performance:** All tuning decisions should be based on performance on the dev set.

5. MODEL EVALUATION AND VALIDATION
--------------------------------
- **Best Model Selection:** Choose the architecture and hyperparameters that yielded the best performance on the dev set (`X_dev`, `y_dev`).
- **Final Training:** Retrain the best model configuration on the entire training dataset (`X_full_train`, `y_full_train` if you created a dev split from it) for a potentially slightly longer number of epochs (or until early stopping on a small fraction of this full train set if preferred).
- **Final Evaluation (if a true Test Set is available):**
    *   Evaluate the *final retrained model* on the unseen hold-out test set (`X_test`, `y_test`). This gives the most unbiased estimate of generalization performance.
    *   If no true test set, the dev set performance is your best estimate.
- **Analyze:**
    *   Overall performance metrics (MSE, MAE, RMSE, R2).
    *   Learning curves (train loss vs. validation loss over epochs) to check for overfitting/underfitting.
    *   Residual plots (predictions vs. actuals, predictions vs. residuals) to identify systematic errors.
    *   Training time and inference speed.
    *   Model complexity vs. performance trade-off.