# Feature Selection
def get_pred(X_train_nn, X_train_gbm, X_val_nn, X_val_gbm, y_train_nn, y_train_gbm, y_val, nn_params, gbm_params):
    from models.utils.nn import tuned_nn
    from models.utils.gbm import tuned_gbm
    import xgboost as xgb

    # Fit & Predict
    # NN
    nn = tuned_nn(nn_params,
                    X_train_nn, y_train_nn,
                    X_val_nn, y_val)
    prob_nn = nn.predict(X_val_nn, verbose=0)
    # GBM
    gbm = tuned_gbm(gbm_params,
                        X_train_gbm, y_train_gbm,
                        X_val_gbm, y_val)
    dval = xgb.DMatrix(X_val_gbm)
    prob_gbm = gbm.predict(dval)
    return prob_nn[:,0], prob_gbm

# Objective Function
def objective(trial, prob_nn, prob_gbm, y_val):
    # Libraries
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import log_loss
    from sklearn.metrics import brier_score_loss

    # Objective
    w = trial.suggest_float('weight', 0, 1)
    combined_probs = w * prob_nn + (1 - w) * prob_gbm
    #return -average_precision_score(y_val, combined_probs)
    #return log_loss(y_val, combined_probs)
    return brier_score_loss(y_val, combined_probs)


def tune_weights(data, split_dict, nn_params, gbm_params, nn_feat=None, gbm_feat=None, n_trials=100):
    # Libraries
    import numpy as np
    from models.utils.DataProcessing import create_splits
    import optuna

    # Supress Logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    weights = {}
    for r in range(2,8):
        print('Round '+str(r))
        weights[r] = {}
        # Create Data
        # NN
        X_SMTL_nn, y_SMTL_nn = create_splits(data, r, train=True, best_features=nn_feat)
        X_nn, y = create_splits(data, r, train=False, best_features=nn_feat)
        # GBM
        X_SMTL_gbm, y_SMTL_gbm = create_splits(data, r, train=True, best_features=gbm_feat)
        X_gbm, _ = create_splits(data, r, train=False, best_features=gbm_feat)

        # Create Splits
        split_idx = int(split_dict[r] * len(X_nn))
        split_idx_SMTL = np.where((X_SMTL_nn == X_nn[split_idx]).all(axis=1))[0][0]

        # NN
        X_train_nn, y_train_nn = X_SMTL_nn[:split_idx_SMTL], y_SMTL_nn[:split_idx_SMTL]
        X_val_nn, y_val = X_nn[split_idx:], y[split_idx:]
        # GBM
        X_train_gbm, y_train_gbm = X_SMTL_gbm[:split_idx_SMTL], y_SMTL_gbm[:split_idx_SMTL]
        X_val_gbm = X_gbm[split_idx:]

        # Get Predictions
        prob_nn, prob_gbm = get_pred(X_train_nn, X_train_gbm, X_val_nn, X_val_gbm, y_train_nn, y_train_gbm, y_val, nn_params[r], gbm_params[r])

        # Create & Optimize Study
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, prob_nn, prob_gbm, y_val), 
                   n_trials=n_trials)
        
        # Store Weights
        weights[r]['NN'] = study.best_params['weight']
        weights[r]['GBM'] = 1 - study.best_params['weight']
    return weights