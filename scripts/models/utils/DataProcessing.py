"""Utilities for creating train/test data splits with SMOTE resampling."""


def create_splits(data, r, split_idx=None, years_list=False, get_features=False, drop_cols=None):
    """Prepare feature matrix and labels for a given tournament round.

    Encodes categorical columns, optionally drops a subset of features,
    and splits into train/val folds when split_idx is provided. The scaler
    is always fit exclusively on the training portion to prevent data leakage.

    Args:
        data: Full modeling DataFrame including all features and metadata.
        r: Tournament round number used to define the binary outcome.
        split_idx: Integer row index at which to split train and val. When
            None the full array is returned unscaled (used for feature
            name retrieval or year-based slicing in backtesting).
        years_list: If True, also return the year column from the feature array.
        get_features: If True, return only the list of feature column names.
        drop_cols: Optional list of feature column names to exclude before
            scaling. Used to apply per-round, per-model feature subsets
            identified during feature selection.

    Returns:
        If get_features is True, returns a list of column name strings.
        If split_idx is None, returns (X_raw, y) as unscaled numpy arrays
            plus years array when years_list is True.
        Otherwise returns (X_train, X_val, y_train, y_val, scaler) plus
            (years_train, years_val) when years_list is True.
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

    X = data_sub.drop(columns="Outcome")
    y = data_sub["Outcome"]

    if drop_cols:
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    if get_features:
        return list(X.columns)

    X_arr = np.array(X)
    y_arr = np.array(y)

    # Return raw arrays for year-based slicing (backtesting / prediction loops)
    if split_idx is None:
        if years_list:
            return X_arr, y_arr, X_arr[:, 0].copy()
        return X_arr, y_arr

    # Split first, then fit scaler on train fold only
    X_train_raw, X_val_raw = X_arr[:split_idx], X_arr[split_idx:]
    y_train, y_val = y_arr[:split_idx], y_arr[split_idx:]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    if years_list:
        years_train = X_train[:, 0].copy()
        years_val = X_val[:, 0].copy()
        return X_train, X_val, y_train, y_val, scaler, years_train, years_val

    return X_train, X_val, y_train, y_val, scaler


def apply_smote(X_train, y_train):
    """Apply BorderlineSMOTE oversampling followed by TomekLinks undersampling.

    Must be called only on the training fold after the train/val split and
    after scaling, to prevent data leakage into the validation set.

    Args:
        X_train: Scaled training feature array.
        y_train: Training labels.

    Returns:
        Tuple of (X_resampled, y_resampled) after SMOTE and TomekLinks.
    """
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import TomekLinks

    sm = BorderlineSMOTE(random_state=23)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X_res, y_res)
    return X_res, y_res
