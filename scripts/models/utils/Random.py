def evaluate_params(params, predictions, correct_picks, years):
    # Libraries
    import numpy as np
    import pandas as pd
    from models.utils.StandarizePredictions import standarize, create_upset_picks
    from models.utils.MakePicks import predict_bracket
    np.random.seed(0)
    
    r32_alpha, s16_alpha, e8_alpha, f4_alpha, ncg_alpha, winner_alpha = params
    points = []
    
    for year in years:
        test_year = 2021 if year == 2019 else year + 1
        pred_df = pd.DataFrame.from_dict(predictions[test_year])
        
        # Standardize
        pred_df = standarize(pred_df)
        # Apply Alphas
        pred_df = create_upset_picks(pred_df, r32_alpha, s16_alpha, e8_alpha, f4_alpha, ncg_alpha, winner_alpha)

        # Get Expected Points
        points_df = pred_df.copy()
        points_df['R32'] = pred_df['R32'] * 10
        points_df['S16'] = points_df['R32'] + pred_df['S16'] * 20
        points_df['E8'] = points_df['S16'] + pred_df['E8'] * 40
        points_df['F4'] = points_df['E8'] + pred_df['F4'] * 80
        points_df['NCG'] = points_df['F4'] + pred_df['NCG'] * 160
        points_df['Winner'] = points_df['NCG'] + pred_df['Winner'] * 320

        # Make Predictions
        _, point, _ = predict_bracket(points_df, correct_picks[str(test_year)])
        points.append(point)

    return np.mean(points), params

def tune_upset_parameters(predictions, correct_picks, years):
    # Libraries
    import numpy as np
    from joblib import Parallel, delayed
    from itertools import product

    np.random.seed(0)
    best_points = 0
    best_params = None

    # Randomness to Tune (Assuming earlier rounds are more random)
    param_grid = list(product(
        np.linspace(0.0, 0.2, 5),
        np.linspace(0.0, 0.2, 5),
        np.linspace(0.0, 0.2, 5),
        np.linspace(0.0, 0.2, 5),
        np.linspace(0.0, 0.1, 5),
        np.linspace(0.0, 0.1, 5)
    ))
    
    results = Parallel(n_jobs=-1)(delayed(evaluate_params)(params, predictions, correct_picks, years) for params in param_grid)

    for mean_points, params in results:
        if mean_points > best_points:
            best_points = mean_points
            best_params = params

    return dict(zip(['R32', 'S16', 'E8', 'F4', 'NCG', 'Winner'], best_params))