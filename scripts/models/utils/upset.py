# Feature Selection
def get_pred(X_train_nn, X_train_gbm, X_val_nn, X_val_gbm, y_train_nn, y_train_gbm, y_val, nn_params, gbm_params, weights):
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
    prob = weights['NN'] * prob_nn[:,0] + weights['GBM'] * prob_gbm
    return prob

# Objective Function
def objective(trial, r, prob_df, y_val):
    # Libraries
    import numpy as np
    from sklearn.metrics import average_precision_score
    from models.utils.upset_picks import create_upset_picks
    # Objective
    w = trial.suggest_float('alpha', 0, 0.2)
    prob = []
    for year in prob_df['Year'].unique():
        probs = prob_df[prob_df['Year']==year]

        # Create Upsets
        upset_prob = create_upset_picks(probs, w, r)

        # Get Probabilities
        prob.append(upset_prob['Prob'].values.reshape(-1, 1))
    prob = np.vstack(prob)
    if sum(np.isnan(prob)) > 0:
        return float('inf')
    else:
        return -average_precision_score(y_val, prob)


def tune(data, split_dict, nn_params, gbm_params, weights, nn_feat=None, gbm_feat=None, n_trials=100):
     # Libraries
    import numpy as np
    import pandas as pd
    from models.utils.DataProcessing import create_splits
    import optuna

    # Supress Logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    alphas = {}
    for r in range(2,8):
        print('Round '+str(r))
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
        prob = get_pred(X_train_nn, X_train_gbm, X_val_nn, X_val_gbm, y_train_nn, y_train_gbm, y_val, nn_params[r], gbm_params[r], weights[r])

        # Attach Year, Region, Seed
        prob_df = pd.DataFrame({
            'Year':data['Year'].values[split_idx:],
            'Region':data['Region'].values[split_idx:],
            'Seed':data['Seed'].values[split_idx:],
            'Prob':prob
        })

        # Create & Optimize Study
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, r, prob_df, y_val), 
                   n_trials=n_trials)

        # Store Weights
        alphas[r] = study.best_params['alpha']
    return alphas
