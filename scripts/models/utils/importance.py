# Feature Selection
def get_pred(X_train_nn, X_train_gbm, X_val_nn, X_val_gbm, y_train_nn, y_train_gbm, y_val, nn_params, gbm_params):
    from models.utils.nn import tuned_nn
    from models.utils.gbm import tuned_gbm

    # Fit
    # NN
    nn = tuned_nn(nn_params,
                    X_train_nn, y_train_nn,
                    X_val_nn, y_val)
    # GBM
    gbm = tuned_gbm(gbm_params,
                        X_train_gbm, y_train_gbm,
                        X_val_gbm, y_val)
    return nn, gbm

def get_importance(data, split_dict, nn_params, gbm_params, weights):
    # Libraries
    import os
    import numpy as np
    import pandas as pd
    from models.utils.DataProcessing import create_splits
    import shap
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    for r in range(2,8):
        print('Round '+str(r))
        # Create Data
        # NN
        X_SMTL_nn, y_SMTL_nn = create_splits(data, r, train=True)
        X_nn, y = create_splits(data, r, train=False)
        # GBM
        X_SMTL_gbm, y_SMTL_gbm = create_splits(data, r, train=True)
        X_gbm, _ = create_splits(data, r, train=False)

        # Create Splits
        split_idx = int(split_dict[r] * len(X_nn))
        split_idx_SMTL = np.where((X_SMTL_nn == X_nn[split_idx]).all(axis=1))[0][0]

        # NN
        X_train_nn, y_train_nn = X_SMTL_nn[:split_idx_SMTL], y_SMTL_nn[:split_idx_SMTL]
        X_val_nn, y_val = X_nn[split_idx:], y[split_idx:]
        # GBM
        X_train_gbm, y_train_gbm = X_SMTL_gbm[:split_idx_SMTL], y_SMTL_gbm[:split_idx_SMTL]
        X_val_gbm = X_gbm[split_idx:]

        # Train Models
        nn, gbm = get_pred(X_train_nn, X_train_gbm, X_val_nn, X_val_gbm, y_train_nn, y_train_gbm, y_val, nn_params[r], gbm_params[r])

        # Get Permutation Importance
        # NN
        nn_exp = shap.DeepExplainer(nn, X_train_nn)
        nn_shap = nn_exp.shap_values(X_val_nn)[:,:,0]
        nn_import = np.mean(np.abs(nn_shap),axis=0)
        # GBM
        gbm_exp = shap.TreeExplainer(gbm,
                                     X_train_gbm,
                                     feature_perturbation='interventional',
                                     model_output='probability')
        gbm_shap = gbm_exp.shap_values(X_val_gbm)
        gbm_import = np.mean(np.abs(gbm_shap),axis=0)
        # Standardize SHAP
        nn_import = nn_import / np.sum(nn_import)
        gbm_import = gbm_import / np.sum(gbm_import)
        # Weighted
        weight_shap = nn_shap*weights[r]['NN'] + gbm_shap*weights[r]['GBM']
        weight_import = nn_import*weights[r]['NN'] + gbm_import*weights[r]['GBM']
        weight_import = weight_import / np.sum(weight_import)
        # Feature Names
        features = create_splits(data, r, train=False, get_features=True)

        # To DF
        # NN
        nn_df = pd.DataFrame({
            "Feature": features,
            "Importance": nn_import,
            "SHAP": np.mean(nn_shap,axis=0)
        })
        # GBM
        gbm_df = pd.DataFrame({
            "Feature": features,
            "Importance": gbm_import,
            "SHAP": np.mean(gbm_shap,axis=0)
        })
        # Weighted Average
        weight_df = pd.DataFrame({
            "Feature": features,
            "Importance": weight_import,
            "SHAP": np.mean(weight_shap,axis=0)
        })

        # Sort
        nn_df.sort_values(by="Importance", ascending=False, inplace=True)
        gbm_df.sort_values(by="Importance", ascending=False, inplace=True)
        weight_df.sort_values(by="Importance", ascending=False, inplace=True)

        # To CSV
        nn_df.to_csv(os.path.join(os.path.abspath(os.getcwd()), f"results/perm_importance/nn/round_{r}.csv"), index=False)
        gbm_df.to_csv(os.path.join(os.path.abspath(os.getcwd()), f"results/perm_importance/gbm/round_{r}.csv"), index=False)
        weight_df.to_csv(os.path.join(os.path.abspath(os.getcwd()), f"results/perm_importance/weighted/round_{r}.csv"), index=False)