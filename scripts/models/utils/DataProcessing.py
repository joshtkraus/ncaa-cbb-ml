"""Utilities for creating train/test data splits with SMOTE resampling."""

# Maps each tournament round to the one grouped-average prefix that is
# semantically matched to that round. All other prefixes are redundant
# (they measure a team relative to opponents it has already faced or will
# never face at that stage) and are dropped before modelling to avoid the
# correlated-feature problem that causes permutation importance to
# systematically undervalue features.
_ROUND_PREFIX = {2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "NCG", 7: "Winner"}

# Seed threshold above which a winning team is considered an upset in each
# round. Teams with seed > threshold who won are upweighted in the loss to
# incentivise the model to identify upsets.
_UPSET_SEED_THRESHOLD = {2: 8, 3: 4, 4: 2, 5: 1, 6: 1, 7: 1}

# Multiplier applied to upset winners in the weighted Brier score objective.
_UPSET_WEIGHT = 2.0

_ALL_PREFIXES = ["R32", "S16", "E8", "F4", "NCG", "Winner"]

_ACTUAL_SUFFIXES = ["Full", "12", "6"]

_KENPOM_COLS = [
    "Tempo",
    "RankTempo",
    "AdjTempo",
    "RankAdjTempo",
    "OE",
    "RankOE",
    "AdjOE",
    "RankAdjOE",
    "DE",
    "RankDE",
    "AdjDE",
    "RankAdjDE",
    "AdjEM",
    "RankAdjEM",
    "Off_1",
    "RankOff_1",
    "Off_2",
    "RankOff_2",
    "Off_3",
    "RankOff_3",
    "Def_1",
    "RankDef_1",
    "Def_2",
    "RankDef_2",
    "Def_3",
    "RankDef_3",
    "Size",
    "SizeRank",
    "Hgt5",
    "Hgt5Rank",
    "Hgt4",
    "Hgt4Rank",
    "Hgt3",
    "Hgt3Rank",
    "Hgt2",
    "Hgt2Rank",
    "Hgt1",
    "Hgt1Rank",
    "HgtEff",
    "HgtEffRank",
    "Exp",
    "ExpRank",
    "Bench",
    "BenchRank",
    "Pts5",
    "Pts5Rank",
    "Pts4",
    "Pts4Rank",
    "Pts3",
    "Pts3Rank",
    "Pts2",
    "Pts2Rank",
    "Pts1",
    "Pts1Rank",
    "OR5",
    "OR5Rank",
    "OR4",
    "OR4Rank",
    "OR3",
    "OR3Rank",
    "OR2",
    "OR2Rank",
    "OR1",
    "OR1Rank",
    "DR5",
    "DR5Rank",
    "DR4",
    "DR4Rank",
    "DR3",
    "DR3Rank",
    "DR2",
    "DR2Rank",
    "DR1",
    "DR1Rank",
]


def _mismatched_avg_cols(r):
    """Return grouped-average and Actual column names for future rounds relative to round r.

    Only prefixes up to and including the matched prefix for round r are kept —
    these represent historical context the model could legitimately have at that
    stage. Future-round prefixes are dropped since those games have not yet
    been played and including them would be conceptually invalid.

    Args:
        r: Tournament round number (2-7).

    Returns:
        List of column name strings to drop.
    """
    # Keep the matched prefix and all earlier-round prefixes; drop future-round prefixes only
    matched = _ROUND_PREFIX[r]
    matched_idx = _ALL_PREFIXES.index(matched)
    future_prefixes = _ALL_PREFIXES[matched_idx + 1 :]
    kenpom_drop = [f"{p}_{c}_Avg" for p in future_prefixes for c in _KENPOM_COLS]
    actual_drop = [f"{p}_Actual_{s}" for p in future_prefixes for s in _ACTUAL_SUFFIXES]
    return kenpom_drop + actual_drop


def create_splits(data, r, val_start=None, get_features=False, drop_cols=None):
    """Prepare feature matrix and labels for a given tournament round.

    Encodes categorical columns, retains Year as a predictor, removes the five
    non-round-matched grouped average prefixes to reduce correlated features,
    and optionally drops a caller-supplied feature subset. When val_start is
    provided the data is split into train (Year < val_start) and val
    (Year >= val_start) folds; the scaler is always fit exclusively on the
    training portion to prevent data leakage.

    Args:
        data: Full modeling DataFrame including all features and metadata.
        r: Tournament round number used to define the binary outcome.
        val_start: Integer year at which the validation set begins. Rows with
            Year >= val_start form the val fold; earlier rows form the train
            fold. When None the full unscaled array is returned (used by
            backtesting and prediction loops that handle splitting themselves).
        get_features: If True, return only the list of feature column names.
        drop_cols: Optional list of additional feature column names to exclude
            before scaling. Used to apply per-round, per-model feature subsets
            identified during feature selection.

    Returns:
        If get_features is True, returns a list of column name strings.
        If val_start is None, returns (X_raw, y_raw, years_raw) as unscaled
            numpy arrays, where years_raw is a copy of the Year column used
            for year-based slicing by callers (Year is also retained in X).
        Otherwise returns (X_train, X_val, y_train, y_val, scaler).
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    mod_data = data.copy()
    mod_data["Outcome"] = 0
    mod_data.loc[mod_data["Round"] < r, "Outcome"] = 0
    mod_data.loc[mod_data["Round"] >= r, "Outcome"] = 1

    data_sub = mod_data.drop(columns=["Team", "Round"])
    data_sub = pd.concat([data_sub, pd.get_dummies(data_sub["Conf"], prefix="Conf")], axis=1)
    data_sub.drop(columns="Conf", inplace=True)
    data_sub = pd.concat([data_sub, pd.get_dummies(data_sub["Region"], prefix="Region")], axis=1)
    data_sub.drop(columns="Region", inplace=True)

    years = np.array(data_sub["Year"])
    y = data_sub["Outcome"]
    X = data_sub.drop(columns=["Outcome"])

    # Drop the five non-round-matched grouped average prefixes
    to_drop = set(_mismatched_avg_cols(r))
    if drop_cols:
        to_drop.update(drop_cols)
    X = X.drop(columns=[c for c in to_drop if c in X.columns])

    if get_features:
        return list(X.columns)

    X_arr = np.array(X)
    y_arr = np.array(y)

    if val_start is None:
        return X_arr, y_arr, years

    train_mask = years < val_start
    val_mask = years >= val_start

    X_train_raw, X_val_raw = X_arr[train_mask], X_arr[val_mask]
    y_train, y_val = y_arr[train_mask], y_arr[val_mask]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    return X_train, X_val, y_train, y_val, scaler


def apply_smote(X_train, y_train):
    """Apply BorderlineSMOTE oversampling followed by TomekLinks undersampling.

    Must be called only on the training fold after the train/val split and
    after scaling, to prevent data leakage into the validation set.

    Falls back to returning the original arrays unchanged if there are too
    few minority class samples to support BorderlineSMOTE (requires at least
    k+1=6 minority samples by default).

    Args:
        X_train: Scaled training feature array.
        y_train: Training labels.

    Returns:
        Tuple of (X_resampled, y_resampled) after SMOTE and TomekLinks,
        or the original arrays if SMOTE cannot be applied.
    """
    import numpy as np
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import TomekLinks

    k_neighbors = 5
    n_minority = int(np.sum(y_train == 1))
    if n_minority <= k_neighbors:
        print(f"      Skipping SMOTE: only {n_minority} minority samples (need >{k_neighbors})")
        return X_train, y_train

    sm = BorderlineSMOTE(random_state=23)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X_res, y_res)
    return X_res, y_res


def create_fold_splits(data, r, fold, drop_cols=None):
    """Prepare scaled train/val arrays for a single walk-forward CV fold.

    Applies the same feature construction as create_splits but uses
    explicit train_years and val_years from a fold definition rather than
    a val_start year. The scaler is fit exclusively on training rows.

    Args:
        data: Full modeling DataFrame.
        r: Tournament round number used to define the binary outcome and
            round-matched feature prefix.
        fold: Dict with keys 'train_years' and 'val_years' as returned by
            make_folds().
        drop_cols: Optional list of additional feature column names to exclude.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val) as scaled numpy arrays.
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    mod_data = data.copy()
    mod_data["Outcome"] = 0
    mod_data.loc[mod_data["Round"] < r, "Outcome"] = 0
    mod_data.loc[mod_data["Round"] >= r, "Outcome"] = 1

    data_sub = mod_data.drop(columns=["Team", "Round"])
    data_sub = pd.concat([data_sub, pd.get_dummies(data_sub["Conf"], prefix="Conf")], axis=1)
    data_sub.drop(columns="Conf", inplace=True)
    data_sub = pd.concat([data_sub, pd.get_dummies(data_sub["Region"], prefix="Region")], axis=1)
    data_sub.drop(columns="Region", inplace=True)

    years = np.array(data_sub["Year"])
    y = data_sub["Outcome"]
    X = data_sub.drop(columns=["Outcome"])

    to_drop = set(_mismatched_avg_cols(r))
    if drop_cols:
        to_drop.update(drop_cols)
    X = X.drop(columns=[c for c in to_drop if c in X.columns])

    X_arr = np.array(X)
    y_arr = np.array(y)

    train_mask = np.isin(years, fold["train_years"])
    val_mask = np.isin(years, fold["val_years"])

    X_train_raw, X_val_raw = X_arr[train_mask], X_arr[val_mask]
    y_train, y_val = y_arr[train_mask], y_arr[val_mask]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    return X_train, X_val, y_train, y_val


def get_class_weights(y_train):
    """Compute per-sample class weights from training labels using sklearn.

    Uses sklearn's 'balanced' strategy, which sets class weights inversely
    proportional to class frequencies:

        weight_c = n_samples / (n_classes * n_samples_c)

    Weights are computed from the original pre-SMOTE labels so they reflect
    the true class imbalance in the training window. The returned array is
    aligned to the (potentially post-SMOTE) y array passed in, mapping each
    sample's label to its corresponding class weight. This means synthetic
    SMOTE samples receive the same weight as the real minority samples they
    were interpolated from, which is consistent with their purpose of
    up-representing the minority class.

    Args:
        y_train: Training label array (may be post-SMOTE). Class weights are
            derived from the unique classes and their frequencies present in
            this array.

    Returns:
        Numpy array of per-sample weights aligned to y_train.
    """
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights, strict=False))
    return np.array([class_weight_dict[label] for label in y_train])


def get_sample_weights(data, r, years_mask):
    """Build a sample weight array upweighting upset winners for a given round.

    Upset winners are defined as teams with seed > _UPSET_SEED_THRESHOLD[r]
    whose outcome is 1 (they won). All other samples receive weight 1.0.

    Args:
        data: Full modeling DataFrame containing 'Seed', 'Round', and 'Year'
            columns.
        r: Tournament round number used to look up the upset seed threshold.
        years_mask: Boolean array aligned to data rows selecting the rows
            relevant to this fold or split.

    Returns:
        Numpy array of sample weights aligned to the masked rows.
    """
    import numpy as np

    subset = data[years_mask].copy()
    subset["Outcome"] = (subset["Round"] >= r).astype(int)

    threshold = _UPSET_SEED_THRESHOLD[r]
    is_upset_winner = (subset["Seed"] > threshold) & (subset["Outcome"] == 1)

    weights = np.ones(len(subset))
    weights[is_upset_winner.values] = _UPSET_WEIGHT
    return weights
