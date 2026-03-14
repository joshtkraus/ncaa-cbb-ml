"""Walk-forward cross-validation fold generation for hyperparameter tuning."""

# Number of folds
_N_FOLDS = 3


def make_folds(data, n_folds=_N_FOLDS):
    """Generate walk-forward CV fold definitions from available years.

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

    # Assign each year to a fold index
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
