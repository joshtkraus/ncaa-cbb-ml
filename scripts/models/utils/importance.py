"""SHAP-based permutation importance computation for ensemble models."""


def get_pred(
    X_train_nn,
    X_train_gbm,
    X_val_nn,
    X_val_gbm,
    y_train_nn,
    y_train_gbm,
    y_val,
    nn_params,
    gbm_params,
):
    """Fit NN and GBM models on training data for importance evaluation.

    Args:
        X_train_nn: Training features for the NN.
        X_train_gbm: Training features for the GBM.
        X_val_nn: Validation features for the NN.
        X_val_gbm: Validation features for the GBM.
        y_train_nn: Training labels for the NN.
        y_train_gbm: Training labels for the GBM.
        y_val: Validation labels shared by both models.
        nn_params: Tuned NN hyperparameter dict.
        gbm_params: Tuned GBM hyperparameter dict.

    Returns:
        Tuple of (fitted NN model, fitted GBM model).
    """
    from models.utils.gbm import tuned_gbm
    from models.utils.nn import tuned_nn

    nn = tuned_nn(nn_params, X_train_nn, y_train_nn, X_val_nn, y_val)
    gbm = tuned_gbm(gbm_params, X_train_gbm, y_train_gbm, X_val_gbm, y_val)
    return nn, gbm


def get_importance(data, split_dict, nn_params, gbm_params, weights):
    """Compute and export SHAP-based feature importance for all rounds.

    Splits data first, fits the scaler on the training fold only, then applies
    a single SMOTE pass shared by both NN and GBM to prevent data leakage.

    Args:
        data: Full modeling DataFrame.
        split_dict: Dict mapping round number to train/val split ratio.
        nn_params: Dict of tuned NN hyperparameters keyed by round number.
        gbm_params: Dict of tuned GBM hyperparameters keyed by round number.
        weights: Dict of ensemble weights keyed by round number.
    """
    import os
    import warnings

    import numpy as np
    import pandas as pd
    import shap

    from models.utils.DataProcessing import apply_smote, create_splits

    warnings.simplefilter("ignore", UserWarning)

    for r in range(2, 8):
        print("Round " + str(r))

        X_raw, y_raw = create_splits(data, r)
        split_idx = int(split_dict[r] * len(X_raw))
        X_train, X_val, y_train, y_val, _ = create_splits(data, r, split_idx=split_idx)

        # Single SMOTE pass shared by both models
        X_train_res, y_train_res = apply_smote(X_train, y_train)

        nn, gbm = get_pred(
            X_train_res,
            X_train_res,
            X_val,
            X_val,
            y_train_res,
            y_train_res,
            y_val,
            nn_params[r],
            gbm_params[r],
        )

        nn_exp = shap.DeepExplainer(nn, X_train_res)
        nn_shap = nn_exp.shap_values(X_val)[:, :, 0]
        nn_import = np.mean(np.abs(nn_shap), axis=0)

        gbm_exp = shap.TreeExplainer(
            gbm,
            X_train_res,
            feature_perturbation="interventional",
            model_output="probability",
        )
        gbm_shap = gbm_exp.shap_values(X_val)
        gbm_import = np.mean(np.abs(gbm_shap), axis=0)

        nn_import = nn_import / np.sum(nn_import)
        gbm_import = gbm_import / np.sum(gbm_import)

        weight_shap = nn_shap * weights[r]["NN"] + gbm_shap * weights[r]["GBM"]
        weight_import = nn_import * weights[r]["NN"] + gbm_import * weights[r]["GBM"]
        weight_import = weight_import / np.sum(weight_import)

        features = create_splits(data, r, get_features=True)

        nn_df = pd.DataFrame({
            "Feature": features,
            "Importance": nn_import,
            "SHAP": np.mean(nn_shap, axis=0),
        })
        gbm_df = pd.DataFrame({
            "Feature": features,
            "Importance": gbm_import,
            "SHAP": np.mean(gbm_shap, axis=0),
        })
        weight_df = pd.DataFrame({
            "Feature": features,
            "Importance": weight_import,
            "SHAP": np.mean(weight_shap, axis=0),
        })

        nn_df.sort_values(by="Importance", ascending=False, inplace=True)
        gbm_df.sort_values(by="Importance", ascending=False, inplace=True)
        weight_df.sort_values(by="Importance", ascending=False, inplace=True)

        nn_df.to_csv(
            os.path.join(os.path.abspath(os.getcwd()), f"results/perm_importance/nn/round_{r}.csv"),
            index=False,
        )
        gbm_df.to_csv(
            os.path.join(
                os.path.abspath(os.getcwd()), f"results/perm_importance/gbm/round_{r}.csv"
            ),
            index=False,
        )
        weight_df.to_csv(
            os.path.join(
                os.path.abspath(os.getcwd()), f"results/perm_importance/weighted/round_{r}.csv"
            ),
            index=False,
        )
