"""AutoGluon matchup model tuning and fitting for head-to-head prediction."""

import os
import warnings

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings(
    "ignore",
    message="The parameter `force_int_remainder_cols` is deprecated",
    category=FutureWarning,
    module="sklearn",
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"autogluon\.core\.utils\.feature_selection",
)

_AG_TUNING_DIR = "model/autogluon_matchup"
_EXCLUDED_MODELS = ["FASTAI"]
_LEADERBOARD_DIR = "results/model_leaderboard"
_IMPORTANCE_DIR = "results/feature_importance"
DEFAULT_THRESHOLD = 0.5

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


def tune_matchup_autogluon(matchup_data, folds):
    """Tune AutoGluon on walk-forward CV folds for the matchup model.

    Runs two passes over the folds:
      1. Model selection: fits all candidate models to identify the best
         configuration by mean log loss across folds.
      2. Importance pass: re-runs CV with only the winning model config to
         compute feature importances that are attributable to that model alone.

    Args:
        matchup_data: Full matchup DataFrame from build_matchup_dataset.
        folds: List of walk-forward fold dicts from make_folds().

    Returns:
        Dict with 'model_type' and 'hyperparameters'.
    """
    weight_col = "sample_weight"
    cwd = os.path.abspath(os.getcwd())

    # ------------------------------------------------------------------
    # Pass 1: model selection
    # ------------------------------------------------------------------
    model_scores: dict[str, list[float]] = {}
    model_roc_scores: dict[str, list[float]] = {}
    fold_leaderboards = []
    selected_features = []

    print("  [Pass 1/2] Model selection across CV folds...")
    for fold_idx, fold in enumerate(folds):
        print(
            f"    Fold {fold_idx + 1}/{len(folds)}: "
            f"train={fold['train_years']}, val={fold['val_years']}"
        )
        train_mask = np.isin(matchup_data["Year"].to_numpy(), fold["train_years"])
        val_mask = np.isin(matchup_data["Year"].to_numpy(), fold["val_years"])

        train_df = _make_matchup_train_df(matchup_data, train_mask, weight_col)
        val_df = _make_matchup_train_df(matchup_data, val_mask, weight_col).drop(
            columns=[weight_col]
        )

        save_path = os.path.join(cwd, _AG_TUNING_DIR, "selection", f"fold_{fold_idx}")

        predictor = TabularPredictor(
            label="Outcome",
            problem_type="binary",
            eval_metric="log_loss",
            path=save_path,
            sample_weight=weight_col,
            verbosity=0,
        ).fit(
            train_data=train_df,
            tuning_data=val_df,
            presets="best_quality",
            time_limit=1000,
            num_bag_folds=0,
            num_stack_levels=0,
            fit_weighted_ensemble=False,
            excluded_model_types=_EXCLUDED_MODELS,
            feature_prune_kwargs={"force_prune": True, "prune_threshold": 0.0},
        )

        print(len(predictor._trainer.load_model(predictor.model_best).features))
        selected_features.extend(predictor._trainer.load_model(predictor.model_best).features)

        fold_lb = predictor.leaderboard(val_df, extra_metrics=["roc_auc"], silent=True)
        fold_lb["fold"] = fold_idx
        fold_leaderboards.append(fold_lb)

        for _, row in fold_lb.iterrows():
            if "WeightedEnsemble" in row["model"]:
                continue
            model_scores.setdefault(row["model"], []).append(float(row["score_val"]))
            model_roc_scores.setdefault(row["model"], []).append(float(row["roc_auc"]))

    mean_scores = {m: float(np.mean(v)) for m, v in model_scores.items() if len(v) == len(folds)}
    mean_roc_scores = {
        m: float(np.mean(v)) for m, v in model_roc_scores.items() if len(v) == len(folds)
    }

    best_model_name = max(mean_scores, key=lambda m: mean_scores[m])
    print(f"    Best model: {best_model_name}  (mean log loss: {mean_scores[best_model_name]:.4f})")

    all_lb = pd.concat(fold_leaderboards, ignore_index=True)
    numeric_lb_cols = all_lb.select_dtypes(include="number").columns.difference(["fold"])
    avg_leaderboard = all_lb.groupby("model")[numeric_lb_cols].mean().reset_index()
    avg_leaderboard["log_loss_score"] = avg_leaderboard["model"].map(mean_scores)
    avg_leaderboard["roc_auc_score"] = avg_leaderboard["model"].map(mean_roc_scores)
    avg_leaderboard = avg_leaderboard.sort_values("log_loss_score", ascending=False)
    leaderboard_path = os.path.join(cwd, _LEADERBOARD_DIR, "matchup.csv")
    os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)
    avg_leaderboard.to_csv(leaderboard_path, index=False)

    model_type = next(
        (key for prefix, key in _NAME_TO_KEY.items() if best_model_name.startswith(prefix)),
        best_model_name,
    )
    hyperparameters = predictor._trainer.load_model(best_model_name).params
    selected_features = list(set(selected_features))
    ag_params = {
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "selected_features": selected_features,
    }

    # ------------------------------------------------------------------
    # Pass 2: feature importance with the winning model config
    # ------------------------------------------------------------------
    fold_importances = []

    print(f"  [Pass 2/2] Feature importance pass with {best_model_name}...")
    for fold_idx, fold in enumerate(folds):
        print(
            f"    Fold {fold_idx + 1}/{len(folds)}: "
            f"train={fold['train_years']}, val={fold['val_years']}"
        )
        train_mask = np.isin(matchup_data["Year"].to_numpy(), fold["train_years"])
        val_mask = np.isin(matchup_data["Year"].to_numpy(), fold["val_years"])

        train_df = _make_matchup_train_df(matchup_data, train_mask, weight_col)
        val_df = _make_matchup_train_df(matchup_data, val_mask, weight_col).drop(
            columns=[weight_col]
        )

        # Subset to the features selected during the tuning importance pass
        train_df = train_df[ag_params["selected_features"] + ["Outcome", weight_col]]
        val_df = val_df[ag_params["selected_features"] + ["Outcome"]]

        save_path = os.path.join(cwd, _AG_TUNING_DIR, "importance", f"fold_{fold_idx}")

        predictor = TabularPredictor(
            label="Outcome",
            problem_type="binary",
            eval_metric="log_loss",
            path=save_path,
            sample_weight=weight_col,
            verbosity=0,
        ).fit(
            train_data=train_df,
            tuning_data=val_df,
            hyperparameters={ag_params["model_type"]: ag_params["hyperparameters"]},
            num_bag_folds=0,
            num_stack_levels=0,
            fit_weighted_ensemble=False,
        )

        fold_imp = predictor.feature_importance(val_df, num_shuffle_sets=10, silent=True)
        fold_imp["fold"] = fold_idx
        fold_importances.append(fold_imp)

    fold_importances_reset = [
        df.reset_index().rename(columns={"index": "feature"}) for df in fold_importances
    ]
    all_imp = pd.concat(fold_importances_reset, ignore_index=True)
    numeric_imp_cols = all_imp.select_dtypes(include="number").columns.difference(["fold"])
    avg_importance = (
        all_imp.groupby("feature")[numeric_imp_cols]
        .mean()
        .fillna(0)
        .sort_values("importance", ascending=False)
    )
    importance_path = os.path.join(cwd, _IMPORTANCE_DIR, "matchup.csv")
    os.makedirs(os.path.dirname(importance_path), exist_ok=True)
    avg_importance.to_csv(importance_path)

    return ag_params


def fit_matchup_autogluon(matchup_data, train_mask, ag_params, save_path):
    """Fit a single matchup model with frozen hyperparameters on a training window.

    Args:
        matchup_data: Full matchup DataFrame.
        train_mask: Boolean array selecting training rows.
        ag_params: Dict with 'model_type', 'hyperparameters', and 'selected_features' keys.
        save_path: Directory for AutoGluon to save the predictor.

    Returns:
        Fitted TabularPredictor instance.
    """
    weight_col = "sample_weight"
    train_df = _make_matchup_train_df(matchup_data, train_mask, weight_col)

    # Subset to the features selected during the tuning importance pass
    keep_cols = ag_params["selected_features"] + ["Outcome", weight_col]
    train_df = train_df[keep_cols]

    predictor = TabularPredictor(
        label="Outcome",
        problem_type="binary",
        eval_metric="log_loss",
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


def predict_matchup(predictor, pred_df, threshold=DEFAULT_THRESHOLD):
    """Predict the outcome of a single matchup and return the result.

    Args:
        predictor: Fitted matchup TabularPredictor.
        pred_df: Single-row feature DataFrame from make_matchup_pred_df.
        threshold: Probability threshold for Team A win prediction.

    Returns:
        Tuple of (team_a_wins, prob_team_a).
    """
    prob_team_a = float(predictor.predict_proba(pred_df)[1].iloc[0])
    team_a_wins = prob_team_a > threshold
    return team_a_wins, prob_team_a


def _make_matchup_train_df(matchup_data, year_mask, weight_col):
    """Build a labeled, weighted training DataFrame for a year subset.

    Args:
        matchup_data: Full matchup DataFrame.
        year_mask: Boolean array selecting rows.
        weight_col: Name of the sample weight column to add.

    Returns:
        DataFrame with features, Outcome, and weight column.
    """
    subset = matchup_data[year_mask].copy()
    subset = subset.drop(columns=["Team_A", "Team_B"])

    y = subset["Outcome"].to_numpy()
    classes = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    class_weight_map = dict(zip(classes, cw, strict=False))
    subset[weight_col] = subset["Outcome"].map(class_weight_map)

    return subset
