# Create Data Splits
def create_splits(team_data,r):
    # Libraries
    import pandas as pd

    # Create Response
    team_data['Outcome'] = 0
    team_data.loc[team_data['Round']<r,'Outcome'] = 0
    team_data.loc[team_data['Round']>=r,'Outcome'] = 1

    # Cols to Drop
    round_drops_pre = {
        2:['Team','Round'],
        3:['Team','Round'],
        4:['Team','Round'],
        5:['Team','Round'],
        6:['Team','Round'],
        7:['Team','Round','Year','First_Year','Seed','Conf Tourney',
           'RankAdjDE','AdjDE','RankOE','RankAdjOE','RankTempo','RankAdjEM','AdjTempo','RankDE','OE',
           'Def_3','Off_3','RankOff_3','Def_1','RankDef_3','Def_2','Off_2','Off_1','RankOff_1','RankDef_2',
           'R32_Actual_Full','R32_Actual_12','R32_Actual_6',
           'S16_Actual_Full','S16_Actual_12','S16_Actual_6',
           'E8_Actual_Full','E8_Actual_12',
           'F4_Actual_Full',
           'NCG_Actual_12','NCG_Actual_6',
           'Winner_Actual_Full','Winner_Actual_12']
    }

    # Drop Cols
    data_sub = team_data.drop(columns=round_drops_pre[r])

    # Dummy Vars
    if 'Conf' in data_sub.columns:
        # Conference
        data_sub = pd.concat([data_sub, pd.get_dummies(data_sub['Conf'], prefix='Conf')], axis=1)
        data_sub.drop(columns='Conf', inplace=True)
    # Region
    if 'Region' in data_sub.columns:
        data_sub = pd.concat([data_sub, pd.get_dummies(data_sub['Region'], prefix='Region')], axis=1)
        data_sub.drop(columns='Region', inplace=True)
    
    # Drop Any Unwanted Dummy Cols
    round_drops_post = {
        2:[],
        3:[],
        4:[],
        5:[],
        6:[],
        7:['Conf_Mid-Cont','Conf_Pac-10','Conf_MEAC','Conf_Southland','Conf_Pac-12','Conf_MVC',
           'Conf_A-10','Conf_MAC','Conf_A-Sun','Conf_AAC','Conf_AEC','Conf_WAC','Conf_MAAC',
           'Conf_Big South','Conf_Summit','Conf_CAA','Conf_Patriot','Conf_Big Sky','Conf_Big West',
           'Conf_CUSA','Conf_NEC','Conf_Sun Belt','Conf_OVC','Conf_Ivy','Conf_Big East',
           'Conf_SWAC','Conf_Southern','Conf_ACC','Conf_Horizon','Conf_SEC',
           'Region_West','Region_East',]
        }
    data_sub = data_sub.drop(columns=round_drops_post[r])

    # Testing & Training Splits
    X = data_sub.drop(columns='Outcome')
    y = data_sub['Outcome']
    return X, y