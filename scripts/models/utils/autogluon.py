"""AutoGluon-based model tuning and fitting for tournament round prediction."""

import os

# Suppress Ray's FutureWarning about accelerator env var override when num_gpus=0.
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import warnings  # noqa: E402

# Suppress sklearn FutureWarning from AutoGluon's internal ColumnTransformer usage.
# The `force_int_remainder_cols` parameter is deprecated in sklearn 1.7 and has no
# effect — this warning originates inside AutoGluon and cannot be fixed from userland.
warnings.filterwarnings(
    "ignore",
    message="The parameter `force_int_remainder_cols` is deprecated",
    category=FutureWarning,
    module="sklearn",
)

# Directory where AutoGluon saves fitted predictors during tuning.
_AG_TUNING_DIR = "model/autogluon"

# Model types to exclude from the AutoGluon search. FastAI requires a separate
# optional install (autogluon.tabular[fastai]) and fails with an ImportError if absent.
_EXCLUDED_MODELS = ["FASTAI"]

# Output directories for per-round leaderboard and feature importance CSVs.
_LEADERBOARD_DIR = "results/model_leaderboard"
_IMPORTANCE_DIR = "results/feature_importance"



def _make_train_df(data, r, year_mask, weight_col="sample_weight"):
    """Build a labeled, weighted DataFrame for a given round and year subset.

    Constructs features using the same round-matched prefix logic as
    create_splits, adds the binary outcome label, and appends balanced
    class weights as a column so AutoGluon can apply them during fitting.

    Categorical columns (Conf, Region) are left as-is since AutoGluon
    handles categoricals natively. No scaling is applied — AutoGluon
    performs its own internal preprocessing per model type.

    Args:
        data: Full modeling DataFrame.
        r: Round number (2–7).
        year_mask: Boolean array aligned to data rows selecting the subset.
        weight_col: Name of the sample weight column to add.

    Returns:
        DataFrame with features, 'Outcome' label column, and weight column.
    """
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    from models.utils.DataProcessing import _mismatched_avg_cols

    subset = data[year_mask].copy()
    subset["Outcome"] = (subset["Round"] >= r).astype(int)
    subset = subset.drop(columns=["Team", "Round"])

    to_drop = set(_mismatched_avg_cols(r))
    subset = subset.drop(columns=[c for c in to_drop if c in subset.columns])

    y = subset["Outcome"].values
    classes = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    class_weight_map = dict(zip(classes, cw))
    subset[weight_col] = subset["Outcome"].map(class_weight_map)

    return subset


def _make_test_df(data, r, year_mask):
    """Build an unlabeled feature DataFrame for prediction.

    Identical feature construction to _make_train_df but without the
    Outcome label or sample weight column.

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

    For each fold, fits a TabularPredictor using the explicit walk-forward
    train/val split. The val fold is passed as tuning_data so AutoGluon uses
    it for internal early stopping and model selection rather than carving its
    own random split from the training data. Bagging is disabled so that
    tuning_data is respected — bagging ignores tuning_data in favour of its
    own internal CV, which would break the temporal fold structure.

    Val ROC-AUC scores are accumulated per base model across folds. The model
    with the highest mean AUC is selected and its exact hyperparameters
    are extracted for use during backtesting.

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
    fold_leaderboards = []   # leaderboard DataFrame per fold
    fold_importances = []    # feature importance DataFrame per fold

    for fold_idx, fold in enumerate(folds):
        print(
            f"    Fold {fold_idx + 1}/{len(folds)}: "
            f"train={fold['train_years']}, val={fold['val_years']}"
        )
        train_mask = np.isin(data["Year"].values, fold["train_years"])
        val_mask = np.isin(data["Year"].values, fold["val_years"])

        train_df = _make_train_df(data, r, train_mask, weight_col=weight_col)
        # Val needs the Outcome label so AutoGluon can use it for early stopping,
        # but we drop the weight column — val is always unweighted.
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
            time_limit=300,
            num_bag_folds=0,       # disable bagging so tuning_data is respected
            num_stack_levels=0,    # disable stacking — no base models to stack on
            fit_weighted_ensemble=False,
            excluded_model_types=_EXCLUDED_MODELS,
        )

        # Read AUC scores directly from the leaderboard — AutoGluon computes
        # them on tuning_data (val_df_labeled) since eval_metric="roc_auc".
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

    mean_scores = {
        m: float(np.mean(v))
        for m, v in model_scores.items()
        if len(v) == len(folds)
    }
    best_model_name = max(mean_scores, key=mean_scores.get)
    print(f"    Best model: {best_model_name}  (mean AUC: {mean_scores[best_model_name]:.4f})")
    print(
        f"    All models: "
        f"{ {m: f'{s:.4f}' for m, s in sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)} }"
    )

    # Average leaderboard metrics across folds. Models that did not appear in
    # every fold (e.g. failed in one) are excluded from the mean. The AUC
    # score column uses mean_scores which is already averaged across folds.
    import pandas as pd

    cwd = os.path.abspath(os.getcwd())

    all_lb = pd.concat(fold_leaderboards, ignore_index=True)
    numeric_lb_cols = all_lb.select_dtypes(include="number").columns.difference(["fold"])
    avg_leaderboard = (
        all_lb.groupby("model")[numeric_lb_cols].mean().reset_index()
    )
    avg_leaderboard["auc_score"] = avg_leaderboard["model"].map(mean_scores)
    avg_leaderboard = avg_leaderboard.sort_values("auc_score", ascending=False)
    leaderboard_path = os.path.join(cwd, _LEADERBOARD_DIR, f"round_{r}.csv")
    os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)
    avg_leaderboard.to_csv(leaderboard_path, index=False)
    print(f"    Saved averaged leaderboard to {leaderboard_path}")

    # Average feature importance across folds. The importance index is the
    # feature name; stack folds and take the mean of all numeric columns.
    all_imp = pd.concat(fold_importances)
    numeric_imp_cols = all_imp.select_dtypes(include="number").columns.difference(["fold"])
    avg_importance = (
        all_imp.groupby(all_imp.index)[numeric_imp_cols].mean()
        .sort_values("importance", ascending=False)
    )
    importance_path = os.path.join(cwd, _IMPORTANCE_DIR, f"round_{r}.csv")
    os.makedirs(os.path.dirname(importance_path), exist_ok=True)
    avg_importance.to_csv(importance_path)
    print(f"    Saved averaged feature importance to {importance_path}")

    # Derive the AutoGluon hyperparameters key from the best model name.
    # AutoGluon encodes model type in the name prefix (e.g. "LightGBM" -> "GBM",
    # "XGBoost" -> "XGB"). We map the name prefix to the short key used in
    # hyperparameters dicts, then extract the fitted params via the trainer.
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
    # Extract hyperparameters via the internal trainer — more stable than .info().
    hyperparameters = predictor._trainer.load_model(best_model_name).params

    return {
        "model_type": model_type,
        "hyperparameters": hyperparameters,
    }


def fit_autogluon(train_df, test_df, ag_params, save_path, weight_col="sample_weight"):
    """Fit a single AutoGluon model instance with exact frozen hyperparameters.

    Used during backtesting to refit the winning model config on each
    walk-forward training fold without any hyperparameter search.

    Args:
        train_df: Training DataFrame with features, 'Outcome' label, and
            sample weight column (produced by _make_train_df).
        test_df: Test DataFrame with features only (no label).
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
