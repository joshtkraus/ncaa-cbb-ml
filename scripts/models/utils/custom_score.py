import numpy as np

def weighted_brier_loss(y_true, y_pred, seed_weights):
    """
    Custom Brier loss function with penalties for overconfidence in higher seeds.

    Parameters:
    - y_true: Array of true binary outcomes (1 if a team made the round, 0 otherwise).
    - y_pred: Array of predicted probabilities.
    - seed_weights: Array of weights corresponding to team seeds.

    Returns:
    - Weighted Brier loss score.
    """
    # Standard Brier score components
    brier_loss = np.mean((y_true - y_pred) ** 2)
    
    # Overconfidence penalty for incorrect high probabilities
    overconfidence_penalty = np.mean(seed_weights * (y_pred ** 2) * (1 - y_true))
    
    # Combine Brier loss and penalty
    weighted_loss = brier_loss + overconfidence_penalty
    return weighted_loss

# Assign weights based on seed
def calculate_seed_weights(seeds):
    max_seed = 16  # Maximum seed value in the tournament
    return 1 + (max_seed - seeds) / max_seed  # Higher seeds get larger weights


# Define a custom scorer
def custom_brier_scorer(estimator, X, y, seeds):
    # Predict probabilities for the positive class
    y_pred = estimator.predict_proba(X)[:, 1]
    
    # Calculate seed weights
    seed_weights = calculate_seed_weights(seeds)
    
    # Return the custom Brier loss
    return -weighted_brier_loss(y, y_pred, seed_weights)  # Negative for minimization