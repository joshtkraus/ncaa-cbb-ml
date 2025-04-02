# Combine Component Models
def combine_model(data,nn_params,gbm_params,weights,correct_picks,backwards_year=2013):
    print('Combining Models...')
    # Libraries
    import os
    from models.utils.DataProcessing import create_splits
    from models.utils.backwards_test import run_test
    from models.utils.StandarizePredictions import standardize_predict

    # Years to Backwards Test
    years = [*range(backwards_year-1,2024)]
    years.remove(2020)

    # Initialize
    predictions = {}
    for year in years:
        if year == 2019:
            test_year = 2021
        else:
            test_year = year+1
        predictions[test_year] = {}
        predictions[test_year]['Team'] = data.loc[data['Year']==test_year,'Team'].values
        predictions[test_year]['Seed'] = data.loc[data['Year']==test_year,'Seed'].values
        predictions[test_year]['Region'] = data.loc[data['Year']==test_year,'Region'].values

    # Iterate Rounds
    for r in range(2,8):
        # Data Splits
        # NN
        X_SMTL_nn, y_SMTL_nn, years_SMTL_nn = create_splits(data, r, train=True, years_list=True)
        X_nn, y, years_nn = create_splits(data, r, train=False, years_list=True)
        # GBM
        X_SMTL_gbm, y_SMTL_gbm, years_SMTL_gbm = create_splits(data, r, train=True, years_list=True)
        X_gbm, _, years_gbm = create_splits(data, r, train=False, years_list=True)

        # Backwards Testing
        predictions = run_test(
            data,
            X_SMTL_nn,
            y_SMTL_nn,
            X_nn,
            X_SMTL_gbm,
            y_SMTL_gbm,
            X_gbm,
            y,
            nn_params[r],
            gbm_params[r],
            weights[r],
            years,
            r,
            predictions,
            years_SMTL_nn,
            years_nn,
            years_SMTL_gbm,
            years_gbm
        )

    # Standardize Predictions, Make Picks
    points_df, accs_df = standardize_predict(years,predictions,correct_picks)   
 
    # Export
    # Picks Accuracy
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/backwards_test/picks_accuracy.csv')
    accs_df.to_csv(path,index=False)
    # Picks Points
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/backwards_test/picks_points.csv')
    points_df.to_csv(path,index=False)