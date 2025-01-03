# Function to Clean Team Names in KP DF
def clean_KP(df):
    df['Team'] = df['Team'].str.replace('^St\.', 'Saint',regex=True)
    df['Team'] = df['Team'].str.replace("Mount St. Mary's", "Mount Saint Mary's",regex=False)
    df['Team'] = df['Team'].str.replace('St\.', 'State',regex=True)
    df['Team'] = df['Team'].str.replace(';', '',regex=False)
    df['Team'] = df['Team'].str.replace('Grambling State', 'Grambling',regex=False)
    df['Team'] = df['Team'].str.replace('Texas A&M Corpus Chris', 'Texas A&M Corpus Christi',regex=False)
    df['Team'] = df['Team'].str.replace('N.C. State', 'North Carolina State',regex=False)
    df['Team'] = df['Team'].str.replace('Virginia Commonwealth', 'VCU',regex=False)
    df['Team'] = df['Team'].str.replace('Stephen F. Austin', 'Stephen F Austin',regex=False)
    df['Team'] = df['Team'].str.replace('College of Charleston', 'Charleston',regex=False)
    df['Team'] = df['Team'].str.replace('Louisiana Lafayette', 'Louisiana',regex=False)
    return df