"""Walk-forward cross-validation fold generation for hyperparameter tuning."""

# Number of folds for walk-forward CV
_N_FOLDS = 3

# Round groups for shared hyperparameter tuning.
# Within each group, the representative round (lowest) defines the feature
# space via create_splits, and the tuned hyperparameters are applied to all
# rounds in the group.
_ROUND_GROUPS = {
    "early": {"rounds": [2, 3, 4], "feature_round": 3},
    "late": {"rounds": [5, 6, 7], "feature_round": 6},
}


def make_folds(data, n_folds=_N_FOLDS):
    """Generate walk-forward CV fold definitions from available years.

    Splits the sorted unique years in data into n_folds consecutive val
    windows, each preceded by all earlier years as training data. The
    2020 COVID year is excluded automatically since it is absent from the
    dataset.

    Args:
        data: Full modeling DataFrame containing a 'Year' column.
        n_folds: Number of folds (default 5).

    Returns:
        List of dicts, each with keys:
            'train_years': sorted list of training years for this fold.
            'val_years':   sorted list of validation years for this fold.
    """
    import numpy as np

    years = sorted(data["Year"].unique())

    # Assign each year to a fold index, distributing as evenly as possible.
    # The first fold is always skipped since it has no training data, so we
    # create n_folds+1 chunks and drop the first, giving n_folds usable folds.
    splits = np.array_split(years, n_folds + 1)

    folds = []
    for i in range(1, n_folds + 1):
        val_years = list(splits[i])
        train_years = [y for chunk in splits[:i] for y in chunk]
        folds.append({
            "train_years": [int(y) for y in train_years],
            "val_years": [int(y) for y in val_years],
        })

    return folds
