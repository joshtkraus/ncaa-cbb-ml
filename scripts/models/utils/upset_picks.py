def standard_winner(prob_df_upset):
    # Standardize
    Winner_Totals = prob_df_upset['Prob'].sum()
    prob_df_upset['Prob'] = round(prob_df_upset['Prob'] / Winner_Totals, 6)
    return prob_df_upset

def standard_ncg(prob_df_upset):
    # Standardize
    NCG_Totals = prob_df_upset.groupby(['NCG_Group']).agg(Round_6_Sum=('Prob', 'sum')).reset_index()
    prob_df_upset = prob_df_upset.merge(NCG_Totals, on=['NCG_Group'])
    prob_df_upset['Prob'] = round(prob_df_upset['Prob'] / prob_df_upset['Round_6_Sum'], 6)
    prob_df_upset = prob_df_upset.drop(columns=['Round_6_Sum'])
    return prob_df_upset

def standard_f4(prob_df_upset):
    # Standardize
    F4_Totals = prob_df_upset.groupby(['Region']).agg(Round_5_Sum=('Prob', 'sum')).reset_index()
    prob_df_upset = prob_df_upset.merge(F4_Totals, on=['Region'])
    prob_df_upset['Prob'] = round(prob_df_upset['Prob'] / prob_df_upset['Round_5_Sum'], 6)
    prob_df_upset = prob_df_upset.drop(columns=['Round_5_Sum'])
    return prob_df_upset

def standard_e8(prob_df_upset):
    # Standardize
    E8_Totals = prob_df_upset.groupby(['Region','E8_Group']).agg(Round_4_Sum=('Prob', 'sum')).reset_index()
    prob_df_upset = prob_df_upset.merge(E8_Totals, on=['Region','E8_Group'])
    prob_df_upset['Prob'] = round(prob_df_upset['Prob'] / prob_df_upset['Round_4_Sum'], 6)
    prob_df_upset = prob_df_upset.drop(columns=['Round_4_Sum'])
    return prob_df_upset

def standard_s16(prob_df_upset):
    # Standardize
    S16_Totals = prob_df_upset.groupby(['Region','S16_Group']).agg(Round_3_Sum=('Prob', 'sum')).reset_index()
    prob_df_upset = prob_df_upset.merge(S16_Totals, on=['Region','S16_Group'])
    prob_df_upset['Prob'] = round(prob_df_upset['Prob'] / prob_df_upset['Round_3_Sum'], 6)
    prob_df_upset = prob_df_upset.drop(columns=['Round_3_Sum'])
    return prob_df_upset

def standard_r32(prob_df_upset):
     # Standardize
    R32_Totals = prob_df_upset.groupby(['Region','R32_Group']).agg(Round_2_Sum=('Prob', 'sum')).reset_index()
    prob_df_upset = prob_df_upset.merge(R32_Totals, on=['Region','R32_Group'])
    prob_df_upset['Prob'] = round(prob_df_upset['Prob'] / prob_df_upset['Round_2_Sum'], 6)
    prob_df_upset = prob_df_upset.drop(columns=['Round_2_Sum'])
    return prob_df_upset

def create_upset_picks(prob_df, alpha, r):
    # Libraries
    import numpy as np
    import pandas as pd
    # Create Upset Probabilities
    prob_df_upset = prob_df.copy()
    # Get Max's
    if r == 7:
        # Standardize
        prob_df_upset = standard_winner(prob_df_upset)
        # Winner
        winner_max = prob_df_upset['Prob'].max()
    elif r == 6:
        # NCG
        prob_df_upset['NCG_Group'] = ['0']*len(prob_df_upset)
        prob_df_upset.loc[prob_df_upset['Region'].isin(['East','West']),'NCG_Group'] = 'Left'
        prob_df_upset.loc[prob_df_upset['Region'].isin(['South','Midwest']),'NCG_Group'] = 'Right'
        # Standardize
        prob_df_upset = standard_ncg(prob_df_upset)
        ncg_max = prob_df_upset.groupby('NCG_Group')['Prob'].max().reset_index()
    elif r == 5:
        # Standardize
        prob_df_upset = standard_f4(prob_df_upset)
        # F4
        f4_max = prob_df_upset.groupby('Region')['Prob'].max().reset_index()
    elif r == 4:
        # E8
        prob_df_upset['E8_Group'] = [0]*len(prob_df_upset)
        prob_df_upset.loc[prob_df_upset['Seed'].isin([1,16,8,9,4,13,5,12]),'E8_Group'] = 1
        prob_df_upset.loc[prob_df_upset['Seed'].isin([2,15,7,10,3,14,6,11]),'E8_Group'] = 2
        # Standardize
        prob_df_upset = standard_e8(prob_df_upset)
        e8_max = prob_df_upset.groupby(['Region','E8_Group'])['Prob'].max().reset_index()
    elif r == 3:
        # S16
        prob_df_upset['S16_Group'] = [0]*len(prob_df_upset)
        prob_df_upset.loc[prob_df_upset['Seed'].isin([1,16,8,9]),'S16_Group'] = 1
        prob_df_upset.loc[prob_df_upset['Seed'].isin([2,15,7,10]),'S16_Group'] = 2
        prob_df_upset.loc[prob_df_upset['Seed'].isin([3,14,6,11]),'S16_Group'] = 3
        prob_df_upset.loc[prob_df_upset['Seed'].isin([4,13,5,12]),'S16_Group'] = 4
        # Standardize
        prob_df_upset = standard_s16(prob_df_upset)
        s16_max = prob_df_upset.groupby(['Region','S16_Group'])['Prob'].max().reset_index()
    else:
        # R32 Groups
        prob_df_upset['R32_Group'] = [0]*len(prob_df_upset)
        prob_df_upset.loc[prob_df_upset['Seed'].isin([1,16]),'R32_Group'] = 1
        prob_df_upset.loc[prob_df_upset['Seed'].isin([2,15]),'R32_Group'] = 2
        prob_df_upset.loc[prob_df_upset['Seed'].isin([3,14]),'R32_Group'] = 3
        prob_df_upset.loc[prob_df_upset['Seed'].isin([4,13]),'R32_Group'] = 4
        prob_df_upset.loc[prob_df_upset['Seed'].isin([5,12]),'R32_Group'] = 5
        prob_df_upset.loc[prob_df_upset['Seed'].isin([6,11]),'R32_Group'] = 6
        prob_df_upset.loc[prob_df_upset['Seed'].isin([7,10]),'R32_Group'] = 7
        prob_df_upset.loc[prob_df_upset['Seed'].isin([8,9]),'R32_Group'] = 8
        # Standardize
        prob_df_upset = standard_r32(prob_df_upset)
        r32_max = prob_df_upset.groupby(['Region','R32_Group'])['Prob'].max().reset_index()

    # Iterate Rounds
    if r == 7:
        seed = prob_df_upset.loc[np.isclose(prob_df_upset['Prob'], winner_max)==True,'Seed'].values[0]
        # chalk pick
        if seed == 1:
            # Subset Teams
            losers = prob_df_upset.loc[np.isclose(prob_df_upset['Prob'], winner_max)==False,'Prob']
            winner = prob_df_upset.loc[np.isclose(prob_df_upset['Prob'], winner_max)==True,'Prob']
            # Adjust Probabilities
            prob_df_upset.loc[np.isclose(prob_df_upset['Prob'], winner_max)==False,'Prob'] = losers + alpha * (1-losers)
            prob_df_upset.loc[np.isclose(prob_df_upset['Prob'], winner_max)==True,'Prob'] = winner - alpha * winner
        # Standardize
        prob_df_upset = standard_winner(prob_df_upset)
    elif r == 6:
        for group in ['Left','Right']:
            # Max
            ncg_rg_max = ncg_max.loc[ncg_max['NCG_Group']==group,'Prob'].values[0]
            seed = prob_df_upset.loc[(prob_df_upset['NCG_Group']==group)&(np.isclose(prob_df_upset['Prob'], ncg_rg_max)==True),'Seed'].values[0]
            # chalk pick
            if seed == 1:
                # Subset Teams
                losers = prob_df_upset.loc[(prob_df_upset['NCG_Group']==group)&(np.isclose(prob_df_upset['Prob'], ncg_rg_max)==False),'Prob']
                winner = prob_df_upset.loc[(prob_df_upset['NCG_Group']==group)&(np.isclose(prob_df_upset['Prob'], ncg_rg_max)==True),'Prob']
                # Adjust Probabilities
                prob_df_upset.loc[(prob_df_upset['NCG_Group']==group)&(np.isclose(prob_df_upset['Prob'], ncg_rg_max)==False),'Prob'] = losers + alpha * (1-losers)
                prob_df_upset.loc[(prob_df_upset['NCG_Group']==group)&(np.isclose(prob_df_upset['Prob'], ncg_rg_max)==True),'Prob'] = winner - alpha * winner
        # Standardize
        prob_df_upset = standard_ncg(prob_df_upset)
        # Drop Group Columns
        prob_df_upset = prob_df_upset.drop(columns=['NCG_Group'])
    elif r == 5:
        for region in ['East','West','South','Midwest']:
            # Max
            f4_rg_max = f4_max.loc[f4_max['Region']==region,'Prob'].values[0]
            seed = prob_df_upset.loc[(prob_df_upset['Region']==region)&(np.isclose(prob_df_upset['Prob'], f4_rg_max)==True),'Seed'].values[0]
            # chalk pick
            if seed == 1:
                # Subset Teams
                losers = prob_df_upset.loc[(prob_df_upset['Region']==region)&(np.isclose(prob_df_upset['Prob'], f4_rg_max)==False),'Prob']
                winner = prob_df_upset.loc[(prob_df_upset['Region']==region)&(np.isclose(prob_df_upset['Prob'], f4_rg_max)==True),'Prob']
                # Adjust Probabilities
                prob_df_upset.loc[(prob_df_upset['Region']==region)&(np.isclose(prob_df_upset['Prob'], f4_rg_max)==False),'Prob'] = losers + alpha * (1-losers)
                prob_df_upset.loc[(prob_df_upset['Region']==region)&(np.isclose(prob_df_upset['Prob'], f4_rg_max)==True),'Prob'] = winner - alpha * winner
        # Standardize
        prob_df_upset = standard_f4(prob_df_upset)
    elif r == 4:
        for region in ['East','West','South','Midwest']:
            for group in [1,2]:
                # Max
                e8_rg_gp_max = e8_max.loc[(e8_max['Region']==region)&(e8_max['E8_Group']==group),'Prob'].values[0]
                seed = prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['E8_Group']==group)&(np.isclose(prob_df_upset['Prob'], e8_rg_gp_max)==True),'Seed'].values[0]
                # chalk pick
                if group == seed:
                    # Subset Teams
                    losers = prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['E8_Group']==group)&(np.isclose(prob_df_upset['Prob'], e8_rg_gp_max)==False),'Prob']
                    winner = prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['E8_Group']==group)&(np.isclose(prob_df_upset['Prob'], e8_rg_gp_max)==True),'Prob']
                    # Adjust Probabilities
                    prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['E8_Group']==group)&(np.isclose(prob_df_upset['Prob'], e8_rg_gp_max)==False),'Prob'] = losers + alpha * (1-losers)
                    prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['E8_Group']==group)&(np.isclose(prob_df_upset['Prob'], e8_rg_gp_max)==True),'Prob'] = winner - alpha * winner
        # Standardize
        prob_df_upset = standard_e8(prob_df_upset)
        # Drop Group Columns
        prob_df_upset = prob_df_upset.drop(columns=['E8_Group'])
    elif r == 3:
        for region in ['East','West','South','Midwest']:
            for group in [1,2,3,4]:
                # Max
                s16_rg_gp_max = s16_max.loc[(s16_max['Region']==region)&(s16_max['S16_Group']==group),'Prob'].values[0]
                seed = prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['S16_Group']==group)&(np.isclose(prob_df_upset['Prob'], s16_rg_gp_max)==True),'Seed'].values[0]
                # chalk pick
                if group == seed:
                    # Subset Teams
                    losers = prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['S16_Group']==group)&(np.isclose(prob_df_upset['Prob'], s16_rg_gp_max)==False),'Prob']
                    winner = prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['S16_Group']==group)&(np.isclose(prob_df_upset['Prob'], s16_rg_gp_max)==True),'Prob']
                    # Adjust Probabilities
                    prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['S16_Group']==group)&(np.isclose(prob_df_upset['Prob'], s16_rg_gp_max)==False),'Prob'] = losers + alpha * (1-losers)
                    prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['S16_Group']==group)&(np.isclose(prob_df_upset['Prob'], s16_rg_gp_max)==True),'Prob'] = winner - alpha * winner
        # Standardize
        prob_df_upset = standard_s16(prob_df_upset)
        # Drop Group Columns
        prob_df_upset = prob_df_upset.drop(columns=['S16_Group'])
    else:
        for region in ['East','West','South','Midwest']:
            for group in [1,2,3,4,5,6,7,8]:
                # Max
                r32_rg_gp_max = r32_max.loc[(r32_max['Region']==region)&(r32_max['R32_Group']==group),'Prob'].values[0]
                seed = prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['R32_Group']==group)&(np.isclose(prob_df_upset['Prob'], r32_rg_gp_max)==True),'Seed'].values[0]
                # chalk pick
                if group == seed:
                    # Subset Teams
                    losers = prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['R32_Group']==group)&(np.isclose(prob_df_upset['Prob'], r32_rg_gp_max)==False),'Prob']
                    winner = prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['R32_Group']==group)&(np.isclose(prob_df_upset['Prob'], r32_rg_gp_max)==True),'Prob']
                    # Adjust Probabilities
                    prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['R32_Group']==group)&(np.isclose(prob_df_upset['Prob'], r32_rg_gp_max)==False),'Prob'] = losers + alpha * (1-losers)
                    prob_df_upset.loc[(prob_df_upset['Region']==region)&(prob_df_upset['R32_Group']==group)&(np.isclose(prob_df_upset['Prob'], r32_rg_gp_max)==True),'Prob'] = winner - alpha * winner
        # Standardize
        prob_df_upset = standard_r32(prob_df_upset)
        # Drop Group Columns
        prob_df_upset = prob_df_upset.drop(columns=['R32_Group'])
    return prob_df_upset