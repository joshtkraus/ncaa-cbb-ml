# Create Historical Seed Probabilities

def calc_seed_prob(df):
    # Libraries
    import pandas as pd

    # Initialize
    R32_Full = []
    S16_Full = []
    E8_Full = []
    F4_Full = []
    NCG_Full = []
    Winner_Full = []

    # Iterate Year
    for year in df['Year'].unique():
        # Get Historical Probabilities
        # Count by Round
        counts = df[df['Year']<year].groupby(['Seed','Round']).size().to_frame('Count').reset_index()
        # Total Occurence of each Seed
        total = len(df[df['Year']<year]) / 16
        # Total for each Round
        R32 = counts[counts['Round']>1].groupby('Seed')['Count'].sum().reset_index()
        S16 = counts[counts['Round']>2].groupby('Seed')['Count'].sum().reset_index()
        E8 = counts[counts['Round']>3].groupby('Seed')['Count'].sum().reset_index()
        F4 = counts[counts['Round']>4].groupby('Seed')['Count'].sum().reset_index()
        NCG = counts[counts['Round']>5].groupby('Seed')['Count'].sum().reset_index()
        Winner = counts[counts['Round']>6].groupby('Seed')['Count'].sum().reset_index()
        # Add Year
        R32['Year'] = year
        S16['Year'] = year
        E8['Year'] = year
        F4['Year'] = year
        NCG['Year'] = year
        Winner['Year'] = year
        # Rename Cols
        R32.columns = ['Seed','R32_Actual','Year']
        S16.columns = ['Seed','S16_Actual','Year']
        E8.columns = ['Seed','E8_Actual','Year']
        F4.columns = ['Seed','F4_Actual','Year']
        NCG.columns = ['Seed','NCG_Actual','Year']
        Winner.columns = ['Seed','Winner_Actual','Year']
        # Divide by Total
        R32['R32_Actual'] = R32['R32_Actual'] / total
        S16['S16_Actual'] = S16['S16_Actual'] / total
        E8['E8_Actual'] = E8['E8_Actual'] / total
        F4['F4_Actual'] = F4['F4_Actual'] / total
        NCG['NCG_Actual'] = NCG['NCG_Actual'] / total
        Winner['Winner_Actual'] = Winner['Winner_Actual'] / total
        # Append
        R32_Full.append(R32)
        S16_Full.append(S16)
        E8_Full.append(E8)
        F4_Full.append(F4)
        NCG_Full.append(NCG)
        Winner_Full.append(Winner)

    # Concat DFs
    R32 = pd.concat(R32_Full)
    S16 = pd.concat(S16_Full)
    E8 = pd.concat(E8_Full)
    F4 = pd.concat(F4_Full)
    NCG = pd.concat(NCG_Full)
    Winner = pd.concat(Winner_Full)

    # Join to df Data
    df = df.merge(R32,on=['Year','Seed'],how='left')
    df = df.merge(S16,on=['Year','Seed'],how='left')
    df = df.merge(E8,on=['Year','Seed'],how='left')
    df = df.merge(F4,on=['Year','Seed'],how='left')
    df = df.merge(NCG,on=['Year','Seed'],how='left')
    df = df.merge(Winner,on=['Year','Seed'],how='left')

    # Fill NAs (no occurences) w/ 0
    df = df.fillna(0)

    # Create Indicator for Missing df Seed (2013)
    df['First_Year'] = 0
    df.loc[df['Year']==df['Year'].min(),'First_Year'] = 1
    return df