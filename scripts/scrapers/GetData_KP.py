# Libraries
import os
import numpy as np
import pandas as pd
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import time

# To Check Year Length
def check_year_length(df):
    if len(df) < 64:
        raise ValueError('< 64 Teams, # of Teams is: '+str(len(df)))
    elif len(df) > 64:
        raise ValueError('> 64 Teams, # of Teams is: '+str(len(df)))

# Empty DF for Loop
kp_data = pd.DataFrame()

# Years to scrape (remove 2020 bc Covid)
years = [*range(2013,2025)]
years.remove(2020)

# URLS for each year
kp_urls = {
    2013:'https://web.archive.org/web/20130318221134/http://kenpom.com/',
    2014:'https://web.archive.org/web/20140318100454/http://kenpom.com/',
    2015:'https://web.archive.org/web/20150316212936/http://kenpom.com/',
    2016:'https://web.archive.org/web/20160314134726/http://kenpom.com/',
    2017:'https://web.archive.org/web/20170314003320/https://kenpom.com/',
    2018:'https://web.archive.org/web/20180312220151/https://kenpom.com/',
    2019:'https://web.archive.org/web/20190318193308/https://kenpom.com/',
    2021:'https://web.archive.org/web/20210318195126/https://kenpom.com/',
    2022:'https://web.archive.org/web/20220314095323/https://kenpom.com/',
    2023:'https://web.archive.org/web/20230313035454/https://kenpom.com/',
    2024:'https://web.archive.org/web/20240318160959/https://kenpom.com/'
}

# Teams who made play-in but lost
playin_dict = {
                2024:['Howard','Virginia','Montana St.','Boise St.'],
                2023:['Southeast Missouri St.','Texas Southern','Nevada','Mississippi St.'],
                2022:['Wyoming','Texas A&M; Corpus Chris','Bryant','Rutgers'],
                2021:["Mount St. Mary's",'Michigan St.','Appalachian St.','Wichita St.'],
                2019:['North Carolina Central','Temple','Prairie View A&M;',"St. John's"],
                2018:['LIU Brooklyn','UCLA','Arizona St.','North Carolina Central'],
                2017:['New Orleans','Providence','North Carolina Central','Wake Forest'],
                2016:['Fairleigh Dickinson','Tulsa','Vanderbilt','Southern'],
                2015:['Boise St.','Manhattan','North Florida','BYU'],
                2014:['Iowa','Texas Southern','Xavier',"Mount St. Mary's"],
                2013:['Long Island','Liberty','Middle Tennessee','Boise St.'],
                2012:['California','Lamar','Mississippi Valley St.','Iona'],
                2011:['UAB','Alabama St.','Arkansas Little Rock','USC'],
                2010:['Winthrop'],
                2009:['Alabama St.'],
                2008:['Coppin St.'],
                2007:['Florida A&M'],
                2006:['Hampton'],
                2005:['Alabama A&M'],
                2004:['Lehigh'],
                2003:['Texas Southern'],
                2002:['Alcorn St.']
                }

# Iterate URLS
print('Scraping KenPom...')
for year in years:
    print(year)
    # Wait
    time.sleep(5)

    # Get HTML
    year_kp_url = kp_urls[year]
    year_hp_req = Request(year_kp_url, headers = {'User-Agent': 'Mozilla/5.0'})
    year_kp_html = urlopen(year_hp_req).read()
    year_kp_soup = BeautifulSoup(year_kp_html, features='lxml')
    year_kp_soup.find_all("span", {'seed-nit'})

    # Remove the seed number from teams in the NIT (to filter out later)
    for span in year_kp_soup.find_all("span", class_='seed-nit'):
        span.decompose()

    # Years 2018 and back have different formatting
    if year <= 2018:
        # Get the text for the headers and row data,
        kp_rows = year_kp_soup.findAll('tr',class_ = lambda table_rows: table_rows != "thead")
        kp_team_stats = [[td.getText() for td in kp_rows[i].findAll(['td','th'])] for i in range(len(kp_rows))]
        # Remove 1st 2 headers from row data
        kp_headers = kp_team_stats[4]
        kp_team_stats = kp_team_stats[5:]
        # Remove extra header rows
        kp_team_stats = [sublist for sublist in kp_team_stats if len(sublist) == 21]
        kp_team_stats = np.array(kp_team_stats)
        indices = [0,1,2,3,4,5,7,9,11,13,15,17,19]
        kp_team_final = []
        for i in range(len(kp_team_stats)):
            kp_team_final.append(kp_team_stats[i][indices].tolist())
        # create df
        kp_yeardata = pd.DataFrame(kp_team_final, columns = kp_headers)
    else:
        # Get the text for the headers and row data,
        kp_rows = year_kp_soup.findAll('tr',class_ = lambda table_rows: table_rows != "thead")
        kp_team_stats = [[td.getText() for td in kp_rows[i].findAll(['td','th'],class_ = lambda td: td != 'td-right')] for i in range(len(kp_rows))]
        # Remove 1st 2 headers from row data
        kp_headers = kp_team_stats[4]
        kp_team_stats = kp_team_stats[5:]
        # Create a Pandas DataFrame from the efficiency data retrieved
        kp_yeardata = pd.DataFrame(kp_team_stats, columns = kp_headers)

    # Filter out teams who didn't make tourney (no seed)
    kp_yeardata = kp_yeardata[kp_yeardata['Team'].str.endswith(('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'))]
    # Remove seeds from remaining teams, trim
    kp_yeardata['Team'] = kp_yeardata['Team'].str.replace('[(\d+)]','',regex=True)
    kp_yeardata['Team'] = kp_yeardata['Team'].str.strip()

    # Filter out play in teams
    kp_yeardata = kp_yeardata[~kp_yeardata['Team'].isin(playin_dict[year])]

    # Delete all unncessecary columns, and rename
    kp_yeardata.columns = ['Rk','Team','Conf','W','AdjEM','AdjO','AdjD','AdjT','Luck',
                          'AdjEM','OppO','OppD','Non-Conf SOS']
    kp_yeardata.drop(['Rk','AdjEM','Non-Conf SOS'], axis=1, inplace=True)

    # Remove losses
    kp_yeardata['W'] = kp_yeardata['W'].str[:2]

    # Fill empty rows with NaN, then delete these rows
    kp_yeardata.replace('', np.nan, inplace=True)
    kp_yeardata.dropna(inplace=True)

    # Delete rows with extra header data (labeled as 'Team')
    kp_yeardata = kp_yeardata[kp_yeardata['Team'] != 'Team']

    # Convert Dtypes
    kp_yeardata[['AdjO','AdjD','AdjT','Luck', 'OppO','OppD']] = kp_yeardata[['AdjO','AdjD','AdjT','Luck','OppO','OppD']].astype(float)
    kp_yeardata['W'] = kp_yeardata['W'].astype(int)

    # Specify the year of each team's Pomeroy data
    kp_yeardata['Year'] = year

    # Standardize Naming
    kp_yeardata['Team'] = kp_yeardata['Team'].str.replace('-',' ')

    # Check Year Length
    check_year_length(kp_yeardata)

    # Append sub df to full
    kp_data = pd.concat([kp_data,kp_yeardata], ignore_index=True)

# Export Data
# Get File Path
path = os.path.join(os.path.abspath(os.getcwd()), 'data/raw/kenpom.csv')
# Export df
kp_data.to_csv(path,index=False)