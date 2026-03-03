"""Utilities for creating train/test data splits with SMOTE resampling."""


def create_splits(data, r, train, years_list=False, get_features=False, return_scaler=False):
    """Create scaled train/test splits for a given tournament round.

    Applies BorderlineSMOTE and TomekLinks resampling when train=True.

    Args:
        data: Full modeling DataFrame including all features and metadata.
        r: Tournament round number used to define the binary outcome.
        train: If True, apply SMOTE/Tomek resampling to the returned data.
        years_list: If True, also return the year column from the scaled array.
        get_features: If True, return only the list of feature column names.
        return_scaler: If True, also return the fitted MinMaxScaler.

    Returns:
        Depending on flags, returns some combination of X, y, years, and scaler.
    """
    import numpy as np
    import pandas as pd
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import TomekLinks
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

    if get_features:
        return list(X.columns)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    if train:
        sm = BorderlineSMOTE(random_state=23)
        X, y = sm.fit_resample(X, y)
        tl = TomekLinks()
        X, y = tl.fit_resample(X, y)

    if years_list:
        years_col = X[:, 0]

    X = np.array(X)
    y = np.array(y)

    years = np.array([])
    if years_list:
        years = np.array(years_col)
        if not return_scaler:
            return X, y, years
        return X, y, years, scaler
    else:
        if not return_scaler:
            return X, y
        return X, y, scaler
