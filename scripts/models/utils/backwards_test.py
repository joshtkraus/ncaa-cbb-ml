"""Walk-forward backtesting using a frozen AutoGluon model config per round."""


def run_test(data, ag_params, years, r, predictions):
    """Run walk-forward backtesting for a single round using AutoGluon.

    Args:
        data: Full modeling DataFrame.
        ag_params: Dict keyed by round number, each with 'model_type' and
            'hyperparameters' keys as produced by tune_autogluon.
        years: List of backtest training years to iterate over.
        r: Tournament round number.
        predictions: Dict to store per-year round predictions.

    Returns:
        Updated predictions dict with Round_{r} arrays added for each test year.
    """
    import tempfile

    from models.utils.autogluon import _make_test_df, _make_train_df, fit_autogluon

    params = ag_params[r]

    for year in years:
        test_year = 2021 if year == 2019 else year + 1

        train_mask = data["Year"].values < test_year
        test_mask = data["Year"].values == test_year

        train_df = _make_train_df(data, r, train_mask)
        test_df = _make_test_df(data, r, test_mask)

        with tempfile.TemporaryDirectory() as tmp_dir:
            predictor = fit_autogluon(train_df, params, save_path=tmp_dir)
            prob = predictor.predict_proba(test_df)[1].values

        predictions[test_year]["Round_" + str(r)] = prob

    return predictions
