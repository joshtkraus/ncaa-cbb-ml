# Standardize Predictions by Group
def standarize(df):
    # Libraries
    import pandas as pd

    # Rename Columns
    df.columns = ['Team','Seed','Region','R32','S16','E8','F4','NCG','Winner']
    
    # R32 Groups
    df['R32_Group'] = [0]*len(df)
    df.loc[df['Seed'].isin([1,16]),'R32_Group'] = 1
    df.loc[df['Seed'].isin([2,15]),'R32_Group'] = 2
    df.loc[df['Seed'].isin([3,14]),'R32_Group'] = 3
    df.loc[df['Seed'].isin([4,13]),'R32_Group'] = 4
    df.loc[df['Seed'].isin([5,12]),'R32_Group'] = 5
    df.loc[df['Seed'].isin([6,11]),'R32_Group'] = 6
    df.loc[df['Seed'].isin([7,10]),'R32_Group'] = 7
    df.loc[df['Seed'].isin([8,9]),'R32_Group'] = 8
    R32_Totals = df.groupby(['Region','R32_Group']).agg(Round_2_Sum=('R32', 'sum')).reset_index()

    # S16 Groups
    df['S16_Group'] = [0]*len(df)
    df.loc[df['Seed'].isin([1,16,8,9]),'S16_Group'] = 1
    df.loc[df['Seed'].isin([2,15,7,10]),'S16_Group'] = 2
    df.loc[df['Seed'].isin([3,14,6,11]),'S16_Group'] = 3
    df.loc[df['Seed'].isin([4,13,5,12]),'S16_Group'] = 4
    S16_Totals = df.groupby(['Region','S16_Group']).agg(Round_3_Sum=('S16', 'sum')).reset_index()

    # E8 Groups
    df['E8_Group'] = [0]*len(df)
    df.loc[df['Seed'].isin([1,16,8,9,4,13,5,12]),'E8_Group'] = 1
    df.loc[df['Seed'].isin([2,15,7,10,3,14,6,11]),'E8_Group'] = 2
    E8_Totals = df.groupby(['Region','E8_Group']).agg(Round_4_Sum=('E8', 'sum')).reset_index()

    # F4 Group
    F4_Totals = df.groupby(['Region']).agg(Round_5_Sum=('F4', 'sum')).reset_index()

    # NCG Group
    df['NCG_Group'] = ['0']*len(df)
    df.loc[df['Region'].isin(['East','West']),'NCG_Group'] = 'Left'
    df.loc[df['Region'].isin(['South','Midwest']),'NCG_Group'] = 'Right'
    NCG_Totals = df.groupby(['NCG_Group']).agg(Round_6_Sum=('NCG', 'sum')).reset_index()

    # Winner
    Winner_Totals = df['Winner'].sum()

    # Join to DF
    df = df.merge(R32_Totals, on=['Region','R32_Group'])
    df = df.merge(S16_Totals, on=['Region','S16_Group'])
    df = df.merge(E8_Totals, on=['Region','E8_Group'])
    df = df.merge(F4_Totals, on=['Region'])
    df = df.merge(NCG_Totals, on=['NCG_Group'])

    # Standardize Predictions
    df['R32'] = round(df['R32'] / df['Round_2_Sum'], 6)
    df['S16'] = round(df['S16'] / df['Round_3_Sum'], 6)
    df['E8'] = round(df['E8'] / df['Round_4_Sum'], 6)
    df['F4'] = round(df['F4'] / df['Round_5_Sum'], 6)
    df['NCG'] = round(df['NCG'] / df['Round_6_Sum'], 6)
    df['Winner'] = round(df['Winner'] / Winner_Totals, 6)

    # Keep Only Needed Columns
    df = df[['Team','Seed','Region','R32','S16','E8','F4','NCG','Winner']]
    return df

def create_upset_picks(pred_df, r32_alpha, s16_alpha, e8_alpha, f4_alpha, ncg_alpha, winner_alpha):
    # Libraries
    import numpy as np
    import pandas as pd

    # Create Upset Probabilities
    pred_df_upset = pred_df.copy()
    # Get Max's
    # Winner
    winner_max = pred_df_upset['Winner'].max()
    # NCG
    pred_df_upset['NCG_Group'] = ['0']*len(pred_df_upset)
    pred_df_upset.loc[pred_df_upset['Region'].isin(['East','West']),'NCG_Group'] = 'Left'
    pred_df_upset.loc[pred_df_upset['Region'].isin(['South','Midwest']),'NCG_Group'] = 'Right'
    ncg_max = pred_df_upset.groupby('NCG_Group')['NCG'].max().reset_index()
    # F4
    f4_max = pred_df_upset.groupby('Region')['F4'].max().reset_index()
    # E8
    pred_df_upset['E8_Group'] = [0]*len(pred_df_upset)
    pred_df_upset.loc[pred_df_upset['Seed'].isin([1,16,8,9,4,13,5,12]),'E8_Group'] = 1
    pred_df_upset.loc[pred_df_upset['Seed'].isin([2,15,7,10,3,14,6,11]),'E8_Group'] = 2
    e8_max = pred_df_upset.groupby(['Region','E8_Group'])['E8'].max().reset_index()
    # S16
    pred_df_upset['S16_Group'] = [0]*len(pred_df_upset)
    pred_df_upset.loc[pred_df_upset['Seed'].isin([1,16,8,9]),'S16_Group'] = 1
    pred_df_upset.loc[pred_df_upset['Seed'].isin([2,15,7,10]),'S16_Group'] = 2
    pred_df_upset.loc[pred_df_upset['Seed'].isin([3,14,6,11]),'S16_Group'] = 3
    pred_df_upset.loc[pred_df_upset['Seed'].isin([4,13,5,12]),'S16_Group'] = 4
    s16_max = pred_df_upset.groupby(['Region','S16_Group'])['S16'].max().reset_index()
    # R32 Groups
    pred_df_upset['R32_Group'] = [0]*len(pred_df_upset)
    pred_df_upset.loc[pred_df_upset['Seed'].isin([1,16]),'R32_Group'] = 1
    pred_df_upset.loc[pred_df_upset['Seed'].isin([2,15]),'R32_Group'] = 2
    pred_df_upset.loc[pred_df_upset['Seed'].isin([3,14]),'R32_Group'] = 3
    pred_df_upset.loc[pred_df_upset['Seed'].isin([4,13]),'R32_Group'] = 4
    pred_df_upset.loc[pred_df_upset['Seed'].isin([5,12]),'R32_Group'] = 5
    pred_df_upset.loc[pred_df_upset['Seed'].isin([6,11]),'R32_Group'] = 6
    pred_df_upset.loc[pred_df_upset['Seed'].isin([7,10]),'R32_Group'] = 7
    pred_df_upset.loc[pred_df_upset['Seed'].isin([8,9]),'R32_Group'] = 8
    r32_max = pred_df_upset.groupby(['Region','R32_Group'])['R32'].max().reset_index()

    # Iterate Rounds
    for r in range(2,8):
        if r == 7:
            seed = pred_df_upset.loc[np.isclose(pred_df_upset['Winner'], winner_max)==True,'Seed'].values[0]
            # chalk pick
            if seed == 1:
                # Subset Teams
                losers = pred_df_upset.loc[np.isclose(pred_df_upset['Winner'], winner_max)==False,'Winner']
                winner = pred_df_upset.loc[np.isclose(pred_df_upset['Winner'], winner_max)==True,'Winner']
                # Adjust Probabilities
                pred_df_upset.loc[np.isclose(pred_df_upset['Winner'], winner_max)==False,'Winner'] = losers + winner_alpha * (1-losers)
                pred_df_upset.loc[np.isclose(pred_df_upset['Winner'], winner_max)==True,'Winner'] = winner - winner_alpha * winner
        elif r == 6:
            for group in ['Left','Right']:
                # Max
                ncg_rg_max = ncg_max.loc[ncg_max['NCG_Group']==group,'NCG'].values[0]
                seed = pred_df_upset.loc[(pred_df_upset['NCG_Group']==group)&(np.isclose(pred_df_upset['NCG'], ncg_rg_max)==True),'Seed'].values[0]
                # chalk pick
                if seed == 1:
                    # Subset Teams
                    losers = pred_df_upset.loc[(pred_df_upset['NCG_Group']==group)&(np.isclose(pred_df_upset['NCG'], ncg_rg_max)==False),'NCG']
                    winner = pred_df_upset.loc[(pred_df_upset['NCG_Group']==group)&(np.isclose(pred_df_upset['NCG'], ncg_rg_max)==True),'NCG']
                    # Adjust Probabilities
                    pred_df_upset.loc[(pred_df_upset['NCG_Group']==group)&(np.isclose(pred_df_upset['NCG'], ncg_rg_max)==False),'NCG'] = losers + ncg_alpha * (1-losers)
                    pred_df_upset.loc[(pred_df_upset['NCG_Group']==group)&(np.isclose(pred_df_upset['NCG'], ncg_rg_max)==True),'NCG'] = winner - ncg_alpha * winner
        elif r == 5:
            for region in ['East','West','South','Midwest']:
                # Max
                f4_rg_max = f4_max.loc[f4_max['Region']==region,'F4'].values[0]
                seed = pred_df_upset.loc[(pred_df_upset['Region']==region)&(np.isclose(pred_df_upset['F4'], f4_rg_max)==True),'Seed'].values[0]
                # chalk pick
                if seed == 1:
                    # Subset Teams
                    losers = pred_df_upset.loc[(pred_df_upset['Region']==region)&(np.isclose(pred_df_upset['F4'], f4_rg_max)==False),'F4']
                    winner = pred_df_upset.loc[(pred_df_upset['Region']==region)&(np.isclose(pred_df_upset['F4'], f4_rg_max)==True),'F4']
                    # Adjust Probabilities
                    pred_df_upset.loc[(pred_df_upset['Region']==region)&(np.isclose(pred_df_upset['F4'], f4_rg_max)==False),'F4'] = losers + f4_alpha * (1-losers)
                    pred_df_upset.loc[(pred_df_upset['Region']==region)&(np.isclose(pred_df_upset['F4'], f4_rg_max)==True),'F4'] = winner - f4_alpha * winner
        elif r == 4:
            for region in ['East','West','South','Midwest']:
                for group in [1,2]:
                    # Max
                    e8_rg_gp_max = e8_max.loc[(e8_max['Region']==region)&(e8_max['E8_Group']==group),'E8'].values[0]
                    seed = pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['E8_Group']==group)&(np.isclose(pred_df_upset['E8'], e8_rg_gp_max)==True),'Seed'].values[0]
                    # chalk pick
                    if group == seed:
                        # Subset Teams
                        losers = pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['E8_Group']==group)&(np.isclose(pred_df_upset['E8'], e8_rg_gp_max)==False),'E8']
                        winner = pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['E8_Group']==group)&(np.isclose(pred_df_upset['E8'], e8_rg_gp_max)==True),'E8']
                        # Adjust Probabilities
                        pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['E8_Group']==group)&(np.isclose(pred_df_upset['E8'], e8_rg_gp_max)==False),'E8'] = losers + e8_alpha * (1-losers)
                        pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['E8_Group']==group)&(np.isclose(pred_df_upset['E8'], e8_rg_gp_max)==True),'E8'] = winner - e8_alpha * winner
        elif r == 3:
            for region in ['East','West','South','Midwest']:
                for group in [1,2,3,4]:
                    # Max
                    s16_rg_gp_max = s16_max.loc[(s16_max['Region']==region)&(s16_max['S16_Group']==group),'S16'].values[0]
                    seed = pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['S16_Group']==group)&(np.isclose(pred_df_upset['S16'], s16_rg_gp_max)==True),'Seed'].values[0]
                    # chalk pick
                    if group == seed:
                        # Subset Teams
                        losers = pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['S16_Group']==group)&(np.isclose(pred_df_upset['S16'], s16_rg_gp_max)==False),'S16']
                        winner = pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['S16_Group']==group)&(np.isclose(pred_df_upset['S16'], s16_rg_gp_max)==True),'S16']
                        # Adjust Probabilities
                        pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['S16_Group']==group)&(np.isclose(pred_df_upset['S16'], s16_rg_gp_max)==False),'S16'] = losers + s16_alpha * (1-losers)
                        pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['S16_Group']==group)&(np.isclose(pred_df_upset['S16'], s16_rg_gp_max)==True),'S16'] = winner - s16_alpha * winner
        else:
            for region in ['East','West','South','Midwest']:
                for group in [1,2,3,4,5,6,7,8]:
                    # Max
                    r32_rg_gp_max = r32_max.loc[(r32_max['Region']==region)&(r32_max['R32_Group']==group),'R32'].values[0]
                    seed = pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['R32_Group']==group)&(np.isclose(pred_df_upset['R32'], r32_rg_gp_max)==True),'Seed'].values[0]
                    # chalk pick
                    if group == seed:
                        # Subset Teams
                        losers = pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['R32_Group']==group)&(np.isclose(pred_df_upset['R32'], r32_rg_gp_max)==False),'R32']
                        winner = pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['R32_Group']==group)&(np.isclose(pred_df_upset['R32'], r32_rg_gp_max)==True),'R32']
                        # Adjust Probabilities
                        pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['R32_Group']==group)&(np.isclose(pred_df_upset['R32'], r32_rg_gp_max)==False),'R32'] = losers + r32_alpha * (1-losers)
                        pred_df_upset.loc[(pred_df_upset['Region']==region)&(pred_df_upset['R32_Group']==group)&(np.isclose(pred_df_upset['R32'], r32_rg_gp_max)==True),'R32'] = winner - r32_alpha * winner
    # Drop Group Columns
    pred_df_upset = pred_df_upset.drop(columns=['R32_Group','S16_Group','E8_Group','NCG_Group'])
    # Standardize (Again)
    pred_df_upset = standarize(pred_df_upset)
    return pred_df_upset

# To Plot Calibration Curves
def save_calibration_curve(df_joined):
    # Libraries
    import os
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    # Map Round to Column Name
    col_map = {
        2:'Round_2',
        3:'Round_3',
        4:'Round_4',
        5:'Round_5',
        6:'Round_6',
        7:'Round_7'
    }

    # Iterate Rounds
    for r in [2,3,4,5,6,7]:
        # Outcome
        df_joined['Outcome'] = 0
        df_joined.loc[df_joined['Round']==r,'Outcome'] = 1

        # Calibration Curves
        prob_true, prob_pred = calibration_curve(df_joined['Outcome'], df_joined[col_map[r]], pos_label=1, n_bins=8)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve: Round '+str(r))
        plt.legend()
        # Export
        path = os.path.join(os.path.abspath(os.getcwd()), 'results/calibration_curves/Round_'+str(r)+'.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()

def standardize_predict(years,upset_parameters,predictions,correct_picks):
    # Libraries
    import os
    import json
    import numpy as np
    import pandas as pd
    from models.utils.MakePicks import predict_bracket

    # Standardize Predictions & Export
    # Initialize
    points = {}
    points_upset = {}
    pick_accs = {}
    pick_accs_upset = {}
    pred_raw = pd.DataFrame()
    # Iterate Years
    for year in years:
        if year == 2019:
            test_year = 2021
        else:
            test_year = year+1
        # To DF
        pred_df = pd.DataFrame.from_dict(predictions[test_year])
        # Copy
        pred_df_copy = pred_df.copy()
        # Add Year
        pred_df_copy['Year'] = test_year
        # Combine DF
        pred_raw = pd.concat([pred_raw,pred_df_copy],ignore_index=True)
        
        # Normal Predictions
        # Standardize
        pred_df = standarize(pred_df)
        # Export
        path = os.path.join(os.path.abspath(os.getcwd()), 'results/predictions/'+str(test_year)+'.csv')
        pred_df.to_csv(path,index=False)
        # Get Expected Points
        points_df = pred_df.copy()
        points_df['R32'] = pred_df['R32']*10
        points_df['S16'] = pred_df['R32']*10 + pred_df['S16']*20
        points_df['E8'] = pred_df['R32']*10 + pred_df['S16']*20 + pred_df['E8']*40
        points_df['F4'] = pred_df['R32']*10 + pred_df['S16']*20 + pred_df['E8']*40 + pred_df['F4']*80
        points_df['NCG'] = pred_df['R32']*10 + pred_df['S16']*20 + pred_df['E8']*40 + pred_df['F4']*80 + pred_df['NCG']*160
        points_df['Winner'] = pred_df['R32']*10 + pred_df['S16']*20 + pred_df['E8']*40 + pred_df['F4']*80 + pred_df['NCG']*160 + pred_df['Winner']*320
        # Make Picks
        pick_accs[test_year] = {}
        # Make Predictions
        picks, point, acc = predict_bracket(points_df,
                                            correct_picks[str(test_year)])
        # Save Predictions
        path = os.path.join(os.path.abspath(os.getcwd()), 'results/picks/standard/'+str(test_year)+'.json')
        with open(path, 'w') as f:
            json.dump(picks, f)
        # Store
        points[test_year] = point
        pick_accs[test_year]['R32'] = acc['R32'] / 32
        pick_accs[test_year]['S16'] = acc['S16'] / 16
        pick_accs[test_year]['E8'] = acc['E8'] / 8
        pick_accs[test_year]['F4'] = acc['F4'] / 4
        pick_accs[test_year]['NCG'] = acc['NCG'] / 2
        pick_accs[test_year]['Winner'] = acc['Winner']

        # Create Upset Picks
        pred_upset_df = create_upset_picks(pred_df,
                                           upset_parameters['R32'],
                                           upset_parameters['S16'],
                                           upset_parameters['E8'],
                                           upset_parameters['F4'],
                                           upset_parameters['NCG'],
                                           upset_parameters['Winner'])
        # Get Expected Points
        points_df = pred_upset_df.copy()
        points_df['R32'] = pred_upset_df['R32']*10
        points_df['S16'] = pred_upset_df['R32']*10 + pred_upset_df['S16']*20
        points_df['E8'] = pred_upset_df['R32']*10 + pred_upset_df['S16']*20 + pred_upset_df['E8']*40
        points_df['F4'] = pred_upset_df['R32']*10 + pred_upset_df['S16']*20 + pred_upset_df['E8']*40 + pred_upset_df['F4']*80
        points_df['NCG'] = pred_upset_df['R32']*10 + pred_upset_df['S16']*20 + pred_upset_df['E8']*40 + pred_upset_df['F4']*80 + pred_upset_df['NCG']*160
        points_df['Winner'] = pred_upset_df['R32']*10 + pred_upset_df['S16']*20 + pred_upset_df['E8']*40 + pred_upset_df['F4']*80 + pred_upset_df['NCG']*160 + pred_df['Winner']*320
        # Make Picks
        pick_accs_upset[test_year] = {}
        # Make Predictions
        picks, point, acc = predict_bracket(points_df,
                                            correct_picks[str(test_year)])
        # Save Predictions
        path = os.path.join(os.path.abspath(os.getcwd()), 'results/picks/upset/'+str(test_year)+'.json')
        with open(path, 'w') as f:
            json.dump(picks, f)
        # Store
        points_upset[test_year] = point
        pick_accs_upset[test_year]['R32'] = acc['R32'] / 32
        pick_accs_upset[test_year]['S16'] = acc['S16'] / 16
        pick_accs_upset[test_year]['E8'] = acc['E8'] / 8
        pick_accs_upset[test_year]['F4'] = acc['F4'] / 4
        pick_accs_upset[test_year]['NCG'] = acc['NCG'] / 2
        pick_accs_upset[test_year]['Winner'] = acc['Winner']

    # Calibration Curves
    # Read Historical Data
    path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/data.csv')
    historical = pd.read_csv(path)
    # Join
    df_joined = pred_raw.merge(historical,on=['Team','Seed','Region','Year'])
    # Subset
    df_joined = df_joined[['Year','Team','Seed','Region','Round','Round_2','Round_3','Round_4','Round_5','Round_6','Round_7']]
    # Calibration Curve
    save_calibration_curve(df_joined)

    # Create DFs
    # Standard
    # Points
    points_df = pd.DataFrame([points])
    points_df['Mean'] = points_df.mean(axis=1).iloc[0]
    points_df['SD'] = points_df.std(axis=1).iloc[0]
    # Accuracy
    accs_df = pd.DataFrame(pick_accs).reset_index()
    accs_df.rename(columns={'index': 'Round'}, inplace=True)
    accs_df['Mean'] = accs_df.iloc[:, 1:].mean(axis=1)
    accs_df['Standard Deviation'] = accs_df.iloc[:, 1:-1].std(axis=1)
    # Upset
    # Points
    points_upset_df = pd.DataFrame([points_upset])
    points_upset_df['Mean'] = points_upset_df.mean(axis=1).iloc[0]
    points_upset_df['SD'] = points_upset_df.std(axis=1).iloc[0]
    # Accuracy
    accs_upset_df = pd.DataFrame(pick_accs_upset).reset_index()
    accs_upset_df.rename(columns={'index': 'Round'}, inplace=True)
    accs_upset_df['Mean'] = accs_upset_df.iloc[:, 1:].mean(axis=1)
    
    # Export
    # Picks Accuracy
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/backwards_test/standard/picks_accuracy.csv')
    accs_df.to_csv(path,index=False)
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/backwards_test/upset/picks_accuracy.csv')
    accs_upset_df.to_csv(path,index=False)
    # Picks Points
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/backwards_test/standard/picks_points.csv')
    points_df.to_csv(path,index=False)
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/backwards_test/upset/picks_points.csv')
    points_upset_df.to_csv(path,index=False)