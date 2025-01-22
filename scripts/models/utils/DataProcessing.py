# Create Data Splits
def create_splits(team_data,r):
    # Libraries
    import pandas as pd

    # Create Response
    team_data['Outcome'] = 0
    team_data.loc[team_data['Round']<r,'Outcome'] = 0
    team_data.loc[team_data['Round']>=r,'Outcome'] = 1

    # Cols to Drop
    # round_drops_pre = {
    #     2:['Team','Round','RankOE','RankAdjEM','AdjOE','Off_3'],
    #     3:['Team','Round','RankOE','RankAdjEM','AdjOE','Off_3'],
    #     4:['Team','Round','RankOE','RankAdjEM','AdjOE','Year','Off_3','Seed'],
    #     5:['Team','Round','RankOE','RankAdjEM','AdjOE','Off_3','Seed'],
    #     6:['Team','Round','RankOE','RankAdjEM','AdjOE','Year','Seed'],
    #     7:['Team','Round','RankOE','RankAdjEM','AdjOE','Year','Seed']
    # }
    round_drops_pre = {
        2:['Team','Round'],
        3:['Team','Round'],
        4:['Team','Round'],
        5:['Team','Round'],
        6:['Team','Round'],
        7:['Team','Round']
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
        7:[]
        }
    data_sub = data_sub.drop(columns=round_drops_post[r])

    # Testing & Training Splits
    X = data_sub.drop(columns='Outcome')
    y = data_sub['Outcome']
    return X, y