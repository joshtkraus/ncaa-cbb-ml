def standarize(df):
    # Libraries
    import pandas as pd
    
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
    R32_Totals = df.groupby(['Region','R32_Group']).agg(Round_2_Sum=('Round_2', 'sum')).reset_index()

    # S16 Groups
    df['S16_Group'] = [0]*len(df)
    df.loc[df['Seed'].isin([1,16,8,9]),'S16_Group'] = 1
    df.loc[df['Seed'].isin([2,15,7,10]),'S16_Group'] = 2
    df.loc[df['Seed'].isin([3,14,6,11]),'S16_Group'] = 3
    df.loc[df['Seed'].isin([4,13,5,12]),'S16_Group'] = 4
    S16_Totals = df.groupby(['Region','S16_Group']).agg(Round_3_Sum=('Round_3', 'sum')).reset_index()

    # E8 Groups
    df['E8_Group'] = [0]*len(df)
    df.loc[df['Seed'].isin([1,16,8,9,4,13,5,12]),'E8_Group'] = 1
    df.loc[df['Seed'].isin([2,15,7,10,3,14,6,11]),'E8_Group'] = 2
    E8_Totals = df.groupby(['Region','E8_Group']).agg(Round_4_Sum=('Round_4', 'sum')).reset_index()

    # F4 Group
    F4_Totals = df.groupby(['Region']).agg(Round_5_Sum=('Round_5', 'sum')).reset_index()

    # NCG Group
    df['NCG_Group'] = ['0']*len(df)
    df.loc[df['Region'].isin(['East','West']),'NCG_Group'] = 'Left'
    df.loc[df['Region'].isin(['South','Midwest']),'NCG_Group'] = 'Right'
    NCG_Totals = df.groupby(['NCG_Group']).agg(Round_6_Sum=('Round_6', 'sum')).reset_index()

    # Winner
    Winner_Totals = df['Round_7'].sum()

    # Join to DF
    df = df.merge(R32_Totals, on=['Region','R32_Group'])
    df = df.merge(S16_Totals, on=['Region','S16_Group'])
    df = df.merge(E8_Totals, on=['Region','E8_Group'])
    df = df.merge(F4_Totals, on=['Region'])
    df = df.merge(NCG_Totals, on=['NCG_Group'])

    # Standardize Predictions
    df['R32'] = round(df['Round_2'] / df['Round_2_Sum'], 6)
    df['S16'] = round(df['Round_3'] / df['Round_3_Sum'], 6)
    df['E8'] = round(df['Round_4'] / df['Round_4_Sum'], 6)
    df['F4'] = round(df['Round_5'] / df['Round_5_Sum'], 6)
    df['NCG'] = round(df['Round_6'] / df['Round_6_Sum'], 6)
    df['Winner'] = round(df['Round_7'] / Winner_Totals, 6)

    # Keep Only Needed Columns
    df = df[['Team','Seed','Region','R32','S16','E8','F4','NCG','Winner']]
    return df