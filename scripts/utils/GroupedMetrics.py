def assign_final(row, c, group_cols, second_highest_dict):
    # Build key based on number of grouping columns
    if len(group_cols) == 1:
        key = row[group_cols[0]]
    else:
        key = tuple(row[col] for col in group_cols)
    
    if row[c] == row['group_max']:
        return second_highest_dict.get(key)
    else:
        return row['group_max']
            
def get_grouped_metrics(df):
    # Libraries
    import pandas as pd
    from warnings import simplefilter
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    # Cols to Calcualte
    cols = ['Tempo','RankTempo','AdjTempo','RankAdjTempo','OE','RankOE','AdjOE','RankAdjOE','DE','RankDE','AdjDE',
            'RankAdjDE','AdjEM','RankAdjEM','Off_1','RankOff_1','Off_2','RankOff_2','Off_3','RankOff_3','Def_1',
            'RankDef_1','Def_2','RankDef_2','Def_3','RankDef_3','Size','SizeRank','Hgt5','Hgt5Rank','Hgt4','Hgt4Rank',
            'Hgt3','Hgt3Rank','Hgt2','Hgt2Rank','Hgt1','Hgt1Rank','HgtEff','HgtEffRank','Exp','ExpRank','Bench','BenchRank',
            'Pts5','Pts5Rank','Pts4','Pts4Rank','Pts3','Pts3Rank','Pts2','Pts2Rank','Pts1','Pts1Rank','OR5','OR5Rank',
            'OR4','OR4Rank','OR3','OR3Rank','OR2','OR2Rank','OR1','OR1Rank','DR5','DR5Rank','DR4','DR4Rank','DR3','DR3Rank',
            'DR2','DR2Rank','DR1','DR1Rank']
    
    # Groups
    # R32
    df['R32_Group'] = [0]*len(df)
    df.loc[df['Seed'].isin([1,16]),'R32_Group'] = 1
    df.loc[df['Seed'].isin([2,15]),'R32_Group'] = 2
    df.loc[df['Seed'].isin([3,14]),'R32_Group'] = 3
    df.loc[df['Seed'].isin([4,13]),'R32_Group'] = 4
    df.loc[df['Seed'].isin([5,12]),'R32_Group'] = 5
    df.loc[df['Seed'].isin([6,11]),'R32_Group'] = 6
    df.loc[df['Seed'].isin([7,10]),'R32_Group'] = 7
    df.loc[df['Seed'].isin([8,9]),'R32_Group'] = 8
    # S16
    df['S16_Group'] = [0]*len(df)
    df.loc[df['Seed'].isin([1,16,8,9]),'S16_Group'] = 1
    df.loc[df['Seed'].isin([2,15,7,10]),'S16_Group'] = 2
    df.loc[df['Seed'].isin([3,14,6,11]),'S16_Group'] = 3
    df.loc[df['Seed'].isin([4,13,5,12]),'S16_Group'] = 4
    # E8
    df['E8_Group'] = [0]*len(df)
    df.loc[df['Seed'].isin([1,16,8,9,4,13,5,12]),'E8_Group'] = 1
    df.loc[df['Seed'].isin([2,15,7,10,3,14,6,11]),'E8_Group'] = 2
    # NCG
    df['NCG_Group'] = ['0']*len(df)
    df.loc[df['Region'].isin(['East','West']),'NCG_Group'] = 'Left'
    df.loc[df['Region'].isin(['South','Midwest']),'NCG_Group'] = 'Right'

    # Calculate
    for c in cols:
        # Max
        # R32
        group_cols = ['Year','Region','R32_Group']
        df['group_max'] = df.groupby(group_cols)[c].transform('max')
        second_highest = df.groupby(group_cols)[c].apply(
            lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else None
        )
        second_highest_dict = second_highest.to_dict()
        df['R32_'+c+'_Max'] = df.apply(assign_final, args=(c, group_cols, second_highest_dict), axis=1)
        df.drop(columns=['group_max'], inplace=True)
        # S16
        group_cols = ['Year','Region','S16_Group']
        df['group_max'] = df.groupby(group_cols)[c].transform('max')
        second_highest = df.groupby(group_cols)[c].apply(
            lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else None
        )
        second_highest_dict = second_highest.to_dict()
        df['S16_'+c+'_Max'] = df.apply(assign_final, args=(c, group_cols, second_highest_dict), axis=1)
        df.drop(columns=['group_max'], inplace=True)
        # E8
        group_cols = ['Year','Region','E8_Group']
        df['group_max'] = df.groupby(group_cols)[c].transform('max')
        second_highest = df.groupby(group_cols)[c].apply(
            lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else None
        )
        second_highest_dict = second_highest.to_dict()
        df['E8_'+c+'_Max'] = df.apply(assign_final, args=(c, group_cols, second_highest_dict), axis=1)
        df.drop(columns=['group_max'], inplace=True)
        # F4
        group_cols = ['Year','Region']
        df['group_max'] = df.groupby(group_cols)[c].transform('max')
        second_highest = df.groupby(group_cols)[c].apply(
            lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else None
        )
        second_highest_dict = second_highest.to_dict()
        df['F4_'+c+'_Max'] = df.apply(assign_final, args=(c, group_cols, second_highest_dict), axis=1)
        df.drop(columns=['group_max'], inplace=True)
        # NCG
        group_cols = ['Year','NCG_Group']
        df['group_max'] = df.groupby(group_cols)[c].transform('max')
        second_highest = df.groupby(group_cols)[c].apply(
            lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else None
        )
        second_highest_dict = second_highest.to_dict()
        df['NCG_'+c+'_Max'] = df.apply(assign_final, args=(c, group_cols, second_highest_dict), axis=1)
        df.drop(columns=['group_max'], inplace=True)
        # Winner
        group_cols = ['Year']
        df['group_max'] = df.groupby(group_cols)[c].transform('max')
        second_highest = df.groupby(group_cols)[c].apply(
            lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else None
        )
        second_highest_dict = second_highest.to_dict()
        df['Winner_'+c+'_Max'] = df.apply(assign_final, args=(c, group_cols, second_highest_dict), axis=1)
        df.drop(columns=['group_max'], inplace=True)

        # Get Difference
        if 'Rank' in c:
            df['R32_'+c+'_Max'] = df['R32_'+c+'_Max'] - df[c]
            df['S16_'+c+'_Max'] = df['S16_'+c+'_Max'] - df[c]
            df['E8_'+c+'_Max'] = df['E8_'+c+'_Max'] - df[c]
            df['F4_'+c+'_Max'] = df['F4_'+c+'_Max'] - df[c]
            df['NCG_'+c+'_Max'] = df['NCG_'+c+'_Max'] - df[c]
            df['Winner_'+c+'_Max'] = df['Winner_'+c+'_Max'] - df[c]
        else:
            df['R32_'+c+'_Max'] = df[c] - df['R32_'+c+'_Max']
            df['S16_'+c+'_Max'] = df[c] - df['S16_'+c+'_Max']
            df['E8_'+c+'_Max'] = df[c] - df['E8_'+c+'_Max']
            df['F4_'+c+'_Max'] = df[c] - df['F4_'+c+'_Max']
            df['NCG_'+c+'_Max'] = df[c] - df['NCG_'+c+'_Max']
            df['Winner_'+c+'_Max'] = df[c] - df['Winner_'+c+'_Max']

    # Drop
    df.drop(columns=['R32_Group','S16_Group','E8_Group','NCG_Group'],inplace=True)
    return df