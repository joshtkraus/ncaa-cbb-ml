# Function to Clean Team Names in SR DF
def clean_SR(df):
    import pandas as pd
    pd.options.mode.chained_assignment = None
    df.loc[:, 'Team'] = df['Team'].str.replace('Mount St Marys', "Mount Saint Mary's",regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('^St ', 'Saint ',regex=True)
    df.loc[:, 'Team'] = df['Team'].str.replace('Saint Marys Ca', "Saint Mary's",regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Albany Ny', 'Albany',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Saint Josephs', "Saint Joseph's",regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Saint Johns Ny', "Saint John's",regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Saint Peters', "Saint Peter's",regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Loyola Il', 'Loyola Chicago',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace(' Am', ' A&M',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('North Carolina At', 'North Carolina A&T',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Brigham Young', 'BYU',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('California Irvine', 'UC Irvine',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Southern Methodist', 'SMU',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Alabama Birmingham', 'UAB',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Southern California', 'USC',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('North Carolina Wilmington', 'UNC Wilmington',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('North Carolina Asheville', 'UNC Asheville',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('California Davis', 'UC Davis',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('North Carolina Greensboro', 'UNC Greensboro',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Pennsylvania', 'Penn',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Texas Christian', 'TCU',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Maryland Baltimore County', 'UMBC',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Central Florida', 'UCF',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('California Santa Barbara', 'UC Santa Barbara',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Virginia Commonwealth', 'VCU',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Louisiana State', 'LSU',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('College Of Charleston', 'Charleston',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Louisiana Lafayette', 'Louisiana',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Ucla', 'UCLA',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Miami Fl', 'Miami FL',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Mcneese State', 'McNeese',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('McNeese State', 'McNeese',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Iupui', 'IUPUI',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Depaul', 'DePaul',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Miami Oh', 'Miami OH',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Loyola Md', 'Loyola MD',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Texas El Paso', 'UTEP',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Texas Arlington', 'UT Arlington',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Texas San Antonio', 'UTSA',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Nevada Las Vegas', 'UNLV',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Long Island University', 'Long Island',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Saint Francis Pa', 'Saint Francis',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Southern Illinois Edwardsville', 'SIUE',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('Nebraska Omaha', 'Omaha',regex=False)
    df.loc[:, 'Team'] = df['Team'].str.replace('California San Diego', 'UC San Diego',regex=False)
    return df