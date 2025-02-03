# Tune Models
def train_models(team_data, best_features=None):
    print('Tuning Models...')
    # Libraries
    import numpy as np
    from models.Models import Logistic_Fit, RF_Fit, GB_Fit, NN_Fit
    
    # Initialize
    best_params = {}
    accs = {}
    k_neighbors = {}
    
    # Iterate Rounds
    for r in range(2,8):
        print('Round',r)
        # Initalize
        best_params[r] = {}
        accs[r] = {}
        k_neighbors[r] = []

        if best_features != None:
            # Fit Models
            print('Fitting Logistic...')
            best_params[r]['Log'],smote,accs[r]['Log'] = Logistic_Fit(team_data,r,best_features[r])
            k_neighbors[r].append(smote['k_neighbors'])
            print('Fitting RF...')
            best_params[r]['RF'],smote,accs[r]['RF'] = RF_Fit(team_data,r,best_features[r])
            k_neighbors[r].append(smote['k_neighbors'])
            print('Fitting GB...')
            best_params[r]['GB'],smote,accs[r]['GB'] = GB_Fit(team_data,r,best_features[r])
            k_neighbors[r].append(smote['k_neighbors'])
            print('Fitting NN...')
            best_params[r]['NN'],smote,accs[r]['NN'] = NN_Fit(team_data,r,best_features[r])
            k_neighbors[r].append(smote['k_neighbors'])
        else:
            # Fit Models
            print('Fitting Logistic...')
            best_params[r]['Log'],smote,accs[r]['Log'] = Logistic_Fit(team_data,r)
            k_neighbors[r].append(smote['k_neighbors'])
            print('Fitting RF...')
            best_params[r]['RF'],smote,accs[r]['RF'] = RF_Fit(team_data,r)
            k_neighbors[r].append(smote['k_neighbors'])
            print('Fitting GB...')
            best_params[r]['GB'],smote,accs[r]['GB'] = GB_Fit(team_data,r)
            k_neighbors[r].append(smote['k_neighbors'])
            print('Fitting NN...')
            best_params[r]['NN'],smote,accs[r]['NN'] = NN_Fit(team_data,r)
            k_neighbors[r].append(smote['k_neighbors'])
        
        # Determine Median K Neighbors
        best_params[r]['SMOTE'] = {'k_neighbors':int(np.median(k_neighbors[r]))}

        # Normalize Average Precision
        # Total Performance
        total_perform = sum([accs[r]['Log'],accs[r]['RF'],accs[r]['GB'],accs[r]['NN']])
        # Normalize
        accs[r]['Log'] = accs[r]['Log'] / total_perform
        accs[r]['RF'] = accs[r]['RF'] / total_perform
        accs[r]['GB'] = accs[r]['GB'] / total_perform
        accs[r]['NN'] = accs[r]['NN'] / total_perform

    return best_params, accs

# Combine Component Models
def combine_model(team_data,best_params,model_accs,correct_picks,best_features,backwards_year=2013,validation_year=2017,upset_parameters=None,tune=True):
    print('Combining Models...')
    # Libraries
    import numpy as np
    import pandas as pd
    from models.utils.DataProcessing import create_splits
    from models.utils.ModelPipeline import backwards_test
    from models.utils.Random import tune_upset_parameters
    from models.utils.StandarizePredictions import standardize_predict
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    np.random.seed(0)

    # Years to Backwards Test
    years = [*range(backwards_year-1,2024)]
    years.remove(2020)

    # Initialize
    prec_list = {}
    models = {}
    predictions = {}
    for year in years:
        if year == 2019:
            test_year = 2021
        else:
            test_year = year+1
        predictions[test_year] = {}
        predictions[test_year]['Team'] = team_data.loc[team_data['Year']==test_year,'Team'].values
        predictions[test_year]['Seed'] = team_data.loc[team_data['Year']==test_year,'Seed'].values
        predictions[test_year]['Region'] = team_data.loc[team_data['Year']==test_year,'Region'].values

    # Iterate Rounds
    for r in range(2,8):
        # Initialize
        prec_list[r] = []

        # Data Splits
        X, y = create_splits(team_data,r,best_features[r])

        # Backwards Testing
        models, prec_list, predictions = backwards_test(years,validation_year,r,team_data,X,y,
                                                      model_accs,best_params,prec_list,predictions,models)

    # Tune Upset Parameters
    if tune == True:
        upset_parameters = tune_upset_parameters(predictions,correct_picks,years)

    # Standardize Predictions, Make Picks
    standardize_predict(years,upset_parameters,predictions,correct_picks)

    return models, upset_parameters, prec_list