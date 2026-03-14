"""AutoGluon model tuning and fitting for tournament round prediction."""

import os

# Suppress Ray's FutureWarning
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import warnings  # noqa: E402

# Suppress sklearn FutureWarning from AutoGluon's internal ColumnTransformer usage
warnings.filterwarnings(
    "ignore",
    message="The parameter `force_int_remainder_cols` is deprecated",
    category=FutureWarning,
    module="sklearn",
)

# Directory where AutoGluon saves fitted predictors during tuning
_AG_TUNING_DIR = "model/autogluon"

# Model types to exclude
_EXCLUDED_MODELS = ["FASTAI"]

# Output directories for leaderboard and feature importance
_LEADERBOARD_DIR = "results/model_leaderboard"
_IMPORTANCE_DIR = "results/feature_importance"


def _make_train_df(data, r, year_mask, weight_col="sample_weight"):
    """Build a labeled, weighted DataFrame for a given round and year subset.

    Args:
        data: Full modeling DataFrame.
        r: Round number (2–7).
        year_mask: Boolean array aligned to data rows selecting the subset.
        weight_col: Name of the sample weight column to add.

    Returns:
        DataFrame with features, 'Outcome' label column, and weight column.
    """
    import numpy as np
    from models.utils.DataProcessing import _mismatched_avg_cols
    from sklearn.utils.class_weight import compute_class_weight

    subset = data[year_mask].copy()
    subset["Outcome"] = (subset["Round"] >= r).astype(int)
    subset = subset.drop(columns=["Team", "Round"])

    to_drop = set(_mismatched_avg_cols(r))
    subset = subset.drop(columns=[c for c in to_drop if c in subset.columns])

    y = subset["Outcome"].values
    classes = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    class_weight_map = dict(zip(classes, cw, strict=False))
    subset[weight_col] = subset["Outcome"].map(class_weight_map)

    return subset


def _make_test_df(data, r, year_mask):
    """Build an unlabeled feature DataFrame for prediction.

    Args:
        data: Full modeling DataFrame.
        r: Round number (2–7).
        year_mask: Boolean array aligned to data rows selecting the subset.

    Returns:
        DataFrame with features only (no label, no weight column).
    """
    from models.utils.DataProcessing import _mismatched_avg_cols

    subset = data[year_mask].copy()
    subset = subset.drop(columns=["Team", "Round"])

    to_drop = set(_mismatched_avg_cols(r))
    subset = subset.drop(columns=[c for c in to_drop if c in subset.columns])

    return subset


def tune_autogluon(data, r, folds):
    """Tune AutoGluon on walk-forward CV folds and return the best model's params.

    Args:
        data: Full modeling DataFrame.
        r: Round number (2–7).
        folds: List of walk-forward fold dicts from make_folds().

    Returns:
        Dict with keys:
            'model_type': AutoGluon model type string (e.g. 'XGBoost').
            'hyperparameters': Exact hyperparameter dict for that model instance.
    """
    import numpy as np
    from autogluon.tabular import TabularPredictor

    weight_col = "sample_weight"
    model_scores = {}
    fold_leaderboards = []  # leaderboard DataFrame per fold
    fold_importances = []  # feature importance DataFrame per fold

    for fold_idx, fold in enumerate(folds):
        print(
            f"    Fold {fold_idx + 1}/{len(folds)}: "
            f"train={fold['train_years']}, val={fold['val_years']}"
        )
        train_mask = np.isin(data["Year"].values, fold["train_years"])
        val_mask = np.isin(data["Year"].values, fold["val_years"])

        train_df = _make_train_df(data, r, train_mask, weight_col=weight_col)
        # Val needs the outcome for early stopping, but no weights
        val_df_labeled = _make_train_df(data, r, val_mask, weight_col=weight_col).drop(
            columns=[weight_col]
        )
        save_path = os.path.join(
            os.path.abspath(os.getcwd()),
            _AG_TUNING_DIR,
            f"round_{r}_fold_{fold_idx}",
        )

        predictor = TabularPredictor(
            label="Outcome",
            problem_type="binary",
            eval_metric="roc_auc",
            path=save_path,
            sample_weight=weight_col,
            verbosity=0,
        ).fit(
            train_data=train_df,
            tuning_data=val_df_labeled,
            presets="best_quality",
            time_limit=600,
            num_bag_folds=0,
            num_stack_levels=0,
            fit_weighted_ensemble=False,
            excluded_model_types=_EXCLUDED_MODELS,
        )

        # Read AUC scores from the leaderboard
        fold_lb = predictor.leaderboard(val_df_labeled, silent=True)
        fold_lb["fold"] = fold_idx
        fold_leaderboards.append(fold_lb)

        for _, row in fold_lb.iterrows():
            if "WeightedEnsemble" in row["model"]:
                continue
            model_scores.setdefault(row["model"], []).append(float(row["score_val"]))

        fold_imp = predictor.feature_importance(val_df_labeled, silent=True)
        fold_imp["fold"] = fold_idx
        fold_importances.append(fold_imp)

    mean_scores = {m: float(np.mean(v)) for m, v in model_scores.items() if len(v) == len(folds)}
    best_model_name = max(mean_scores, key=mean_scores.get)
    print(f"    Best model: {best_model_name}  (mean AUC: {mean_scores[best_model_name]:.4f})")

    # Average leaderboard metrics across folds
    import pandas as pd

    cwd = os.path.abspath(os.getcwd())

    all_lb = pd.concat(fold_leaderboards, ignore_index=True)
    numeric_lb_cols = all_lb.select_dtypes(include="number").columns.difference(["fold"])
    avg_leaderboard = all_lb.groupby("model")[numeric_lb_cols].mean().reset_index()
    avg_leaderboard["auc_score"] = avg_leaderboard["model"].map(mean_scores)
    avg_leaderboard = avg_leaderboard.sort_values("auc_score", ascending=False)
    leaderboard_path = os.path.join(cwd, _LEADERBOARD_DIR, f"round_{r}.csv")
    os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)
    avg_leaderboard.to_csv(leaderboard_path, index=False)
    print(f"    Saved averaged leaderboard to {leaderboard_path}")

    # Average feature importance across folds
    all_imp = pd.concat(fold_importances)
    numeric_imp_cols = all_imp.select_dtypes(include="number").columns.difference(["fold"])
    avg_importance = (
        all_imp.groupby(all_imp.index)[numeric_imp_cols]
        .mean()
        .sort_values("importance", ascending=False)
    )
    importance_path = os.path.join(cwd, _IMPORTANCE_DIR, f"round_{r}.csv")
    os.makedirs(os.path.dirname(importance_path), exist_ok=True)
    avg_importance.to_csv(importance_path)
    print(f"    Saved averaged feature importance to {importance_path}")

    # Derive the AutoGluon hyperparameters key from the best model name
    _NAME_TO_KEY = {
        "LightGBM": "GBM",
        "CatBoost": "CAT",
        "XGBoost": "XGB",
        "RandomForest": "RF",
        "ExtraTrees": "XT",
        "KNeighbors": "KNN",
        "LinearModel": "LR",
        "NeuralNetTorch": "NN_TORCH",
        "NeuralNetFastAI": "FASTAI",
        "RealMLP": "REALMLP",
        "TabM": "TABM",
    }
    model_type = next(
        (key for prefix, key in _NAME_TO_KEY.items() if best_model_name.startswith(prefix)),
        best_model_name,  # fallback: use name as-is if prefix not recognised
    )
    # Extract hyperparameters via the internal trainer
    hyperparameters = predictor._trainer.load_model(best_model_name).params

    return {
        "model_type": model_type,
        "hyperparameters": hyperparameters,
    }


def fit_autogluon(train_df, ag_params, save_path, weight_col="sample_weight"):
    """Fit a single AutoGluon model instance with exact frozen hyperparameters.

    Args:
        train_df: Training DataFrame with features, 'Outcome' label, and
            sample weight column (produced by _make_train_df).
        ag_params: Dict with 'model_type' and 'hyperparameters' keys as
            returned by tune_autogluon.
        save_path: Directory path for AutoGluon to save the fitted predictor.
        weight_col: Name of the sample weight column in train_df.

    Returns:
        Fitted TabularPredictor instance.
    """
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(
        label="Outcome",
        problem_type="binary",
        eval_metric="roc_auc",
        path=save_path,
        sample_weight=weight_col,
        verbosity=0,
    ).fit(
        train_data=train_df,
        hyperparameters={ag_params["model_type"]: ag_params["hyperparameters"]},
        num_bag_folds=0,
        num_stack_levels=0,
        fit_weighted_ensemble=False,
    )

    return predictor
