def run_test(
            data,
            X_SMTL_nn,
            y_SMTL_nn,
            X_nn,
            X_SMTL_gbm,
            y_SMTL_gbm,
            X_gbm,
            y,
            nn_params,
            gbm_params,
            weights,
            upset,
            years,
            r,
            predictions,
            predictions_upset,
            years_SMTL_nn,
            years_nn,
            years_SMTL_gbm,
            years_gbm
            ):
    # Libraries
    import numpy as np
    import pandas as pd
    from models.utils.DataProcessing import create_splits
    from models.utils.nn import tuned_nn
    from models.utils.gbm import tuned_gbm
    import xgboost as xgb
    from models.utils.upset_picks import create_upset_picks

    # Scaled Years
    full_years = [*range(data['Year'].min(),data['Year'].max()+1)]
    full_years.remove(2020)    
    _, _, years_scaled = create_splits(data, r, train=False,years_list=True)
    years_scaled = sorted(np.unique(years_scaled))

    # Iterate years
    for year in years:
        # Get Test Year
        if year == 2019:
            test_year = 2021
        else:
            test_year = year+1
        idx = np.where(np.array(full_years)==test_year)[0][0]

        # Create Splits                
        X_train_nn, X_test_nn = X_SMTL_nn[years_SMTL_nn < years_scaled[idx]], X_nn[years_nn == years_scaled[idx]]
        y_train_nn = y_SMTL_nn[years_SMTL_nn < years_scaled[idx]]
        X_train_gbm, X_test_gbm = X_SMTL_gbm[years_SMTL_gbm < years_scaled[idx]], X_gbm[years_gbm == years_scaled[idx]]
        y_train_gbm = y_SMTL_gbm[years_SMTL_gbm < years_scaled[idx]]

        # Create & Fit
        # NN
        nn = tuned_nn(nn_params,
                        X_train_nn, y_train_nn)
        # GBM
        gbm = tuned_gbm(gbm_params,
                        X_train_gbm, y_train_gbm)

        # Create Probabilities
        # NN
        prob_nn = nn.predict(X_test_nn, verbose=0).flatten()
        # GBM
        dtest = xgb.DMatrix(X_test_gbm)
        prob_gbm = gbm.predict(dtest)
        # Combine Probabilities
        y_pred = prob_nn * weights['NN'] + prob_gbm * weights['GBM']
        
        # Store Averaged Results
        predictions[test_year]['Round_'+str(r)] = y_pred

        # Create Upset Picks
        prob_df = pd.DataFrame({
            'Year':data.loc[data['Year']==test_year,'Year'],
            'Region':data.loc[data['Year']==test_year,'Region'],
            'Seed':data.loc[data['Year']==test_year,'Seed'],
            'Prob':y_pred
        })
        upset_prob = create_upset_picks(prob_df, upset, r)
        predictions_upset[test_year]['Round_'+str(r)] = upset_prob['Prob'].values

    return predictions, predictions_upset