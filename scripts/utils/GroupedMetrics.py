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
        # R32
        grouped = df.groupby(['Year','Region','R32_Group'])
        n = grouped[c].transform('count')
        mean = grouped[c].transform('mean')
        df['R32_'+c+'_Avg'] = (mean * n - df[c])/(n-1)
        # S16
        grouped = df.groupby(['Year','Region','S16_Group'])
        n = grouped[c].transform('count')
        mean = grouped[c].transform('mean')
        df['S16_'+c+'_Avg'] = (mean * n - df[c])/(n-1)
        # E8
        grouped = df.groupby(['Year','Region','E8_Group'])
        n = grouped[c].transform('count')
        mean = grouped[c].transform('mean')
        df['E8_'+c+'_Avg'] = (mean * n - df[c])/(n-1)
        # F4
        grouped = df.groupby(['Year','Region','Region'])
        n = grouped[c].transform('count')
        mean = grouped[c].transform('mean')
        df['F4_'+c+'_Avg'] = (mean * n - df[c])/(n-1)
        # NCG
        grouped = df.groupby(['Year','NCG_Group'])
        n = grouped[c].transform('count')
        mean = grouped[c].transform('mean')
        df['NCG_'+c+'_Avg'] = (mean * n - df[c])/(n-1)
        # Winner
        grouped = df.groupby(['Year'])
        n = grouped[c].transform('count')
        mean = grouped[c].transform('mean')
        df['Winner_'+c+'_Avg'] = (mean * n - df[c])/(n-1)

        # Seed
        grouped = df.groupby(['Year','Seed'])
        n = grouped[c].transform('count')
        mean = grouped[c].transform('mean')
        df[c+'_Seed_Avg'] = (mean * n - df[c])/(n-1)

        # Get Difference
        if 'Rank' in c:
            df['R32_'+c+'_Avg'] = df['R32_'+c+'_Avg'] - df[c]
            df['S16_'+c+'_Avg'] = df['S16_'+c+'_Avg'] - df[c]
            df['E8_'+c+'_Avg'] = df['E8_'+c+'_Avg'] - df[c]
            df['F4_'+c+'_Avg'] = df['F4_'+c+'_Avg'] - df[c]
            df['NCG_'+c+'_Avg'] = df['NCG_'+c+'_Avg'] - df[c]
            df['Winner_'+c+'_Avg'] = df['Winner_'+c+'_Avg'] - df[c]
            df[c+'_Seed_Avg'] = df[c+'_Seed_Avg'] - df[c]
        else:
            df['R32_'+c+'_Avg'] = df[c] - df['R32_'+c+'_Avg']
            df['S16_'+c+'_Avg'] = df[c] - df['S16_'+c+'_Avg']
            df['E8_'+c+'_Avg'] = df[c] - df['E8_'+c+'_Avg']
            df['F4_'+c+'_Avg'] = df[c] - df['F4_'+c+'_Avg']
            df['NCG_'+c+'_Avg'] = df[c] - df['NCG_'+c+'_Avg']
            df['Winner_'+c+'_Avg'] = df[c] - df['Winner_'+c+'_Avg']
            df[c+'_Seed_Avg'] = df[c] - df[c+'_Seed_Avg']

    # Drop
    df.drop(columns=['R32_Group','S16_Group','E8_Group','NCG_Group'],inplace=True)
    return df