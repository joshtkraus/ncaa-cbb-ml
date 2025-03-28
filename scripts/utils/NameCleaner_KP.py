# Function to Clean Team Names in KP DF
def clean_KP(df):
    import pandas as pd
    pd.options.mode.chained_assignment = None
    df.loc[:, 'Team'] = df['Team'].str.replace('^St\.', 'Saint',regex=True)
    df.loc[:, 'Team'] = df['Team'].str.replace("Mount St. Mary's", "Mount Saint Mary's",regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('St\.', 'State',regex=True)
    df.loc[:, 'Team'] = df['Team'].str.replace(';', '',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Grambling State', 'Grambling',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Texas A&M Corpus Chris', 'Texas A&M Corpus Christi',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('N.C. State', 'North Carolina State',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Virginia Commonwealth', 'VCU',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Stephen F. Austin', 'Stephen F Austin',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('College of Charleston', 'Charleston',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Louisiana Lafayette', 'Louisiana',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Detroit', 'Detroit Mercy',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Central Connecticut', 'Central Connecticut State',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Troy State', 'Troy',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('^NC Asheville', 'UNC Asheville',regex=True)
    df.loc[:, 'Team'] = df['Team'].str.replace('Brigham Young', 'BYU',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Texas San Antonio', 'UTSA',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Nevada Las Vegas', 'UNLV',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('LIU Brooklyn', 'Long Island',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Southern Miss', 'Southern Mississippi',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Nebraska Omaha', 'Omaha',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('McNeese State', 'McNeese',regex=False)
    return df