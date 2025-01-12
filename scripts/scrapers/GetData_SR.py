# Web Scraper for Sports Reference

# Libraries
import os
import pandas as pd
import ssl
import certifi
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import urlparse
import json

# Unit Tests
def check_year_length_df(df):
    if len(df) < 64:
        raise ValueError('< 64 Teams, # of Teams is: '+str(len(df)))
    elif len(df) > 64:
        raise ValueError('> 64 Teams, # of Teams is: '+str(len(df)))
def check_year_round_length_dict(picks_dict):
    for region, rounds in picks_dict.items():
        if region == 'Winner':
            if type(rounds) != str:
                raise ValueError('Winner Incorrect, # of Teams is: '+str(len(rounds)))
        elif region == 'NCG':
            if len(rounds) != 2:
                raise ValueError('NCG Incorrect, # of Teams is: '+str(len(rounds)))
        else:
            for round, teams in rounds.items():
                if round == 'F4':
                    if len(teams) != 1:
                        raise ValueError(region + ' F4 Incorrect, # of Teams is: '+str(len(teams)))
                elif round == 'E8':
                    if len(teams) != 2:
                        raise ValueError(region + ' E8 Incorrect, # of Teams is: '+str(len(teams)))
                elif round == 'S16':
                    if len(teams) != 4:
                        raise ValueError(region + ' S16 Incorrect, # of Teams is: '+str(len(teams)))
                elif round == 'R32':
                    if len(teams) != 8:
                        raise ValueError(region + ' R32 Incorrect, # of Teams is: '+str(len(teams)))
                else:
                    if len(teams) != 16:
                        raise ValueError(region + ' R64 Incorrect, # of Teams is: '+str(len(teams)))

# Create empty DataFrames to store the data created in the for loop
seeddata = pd.DataFrame()
bracket_data = {}

# Years to scrape
years = [*range(2006,2025)]
years.remove(2020)

# All possible region names
regions = ['east', 'west', 'midwest', 'south', 'southeast', 'southwest','minneapolis','atlanta',
            'oakland','washington','syracuse','albuquerque','austin','chicago','stlouis',
            'eastrutherford','phoenix']
# Standardize
regions_convert = {
                    2024:{'east':'West','west':'East','south':'South','midwest':'Midwest'},
                    2023:{'south':'West','east':'East','midwest':'South','west':'Midwest'},
                    2022:{'west':'West','east':'East','south':'South','midwest':'Midwest'},
                    2021:{'west':'West','east':'East','south':'South','midwest':'Midwest'},
                    2019:{'east':'West','west':'East','south':'South','midwest':'Midwest'},
                    2018:{'south':'West','west':'East','east':'South','midwest':'Midwest'},
                    2017:{'east':'West','west':'East','midwest':'South','south':'Midwest'},
                    2016:{'south':'West','west':'East','east':'South','midwest':'Midwest'},
                    2015:{'midwest':'West','west':'East','east':'South','south':'Midwest'},
                    2014:{'south':'West','east':'East','west':'South','midwest':'Midwest'},
                    2013:{'midwest':'West','west':'East','south':'South','east':'Midwest'},
                    2012:{'south':'West','west':'East','east':'South','midwest':'Midwest'},
                    2011:{'east':'West','west':'East','southwest':'South','southeast':'Midwest'},
                    2010:{'midwest':'West','west':'East','east':'South','south':'Midwest'},
                    2009:{'midwest':'West','west':'East','east':'South','south':'Midwest'},
                    2008:{'east':'West','midwest':'East','south':'South','west':'Midwest'},
                    2007:{'midwest':'West','west':'East','east':'South','south':'Midwest'},
                    2006:{'atlanta':'West','oakland':'East','washington':'South','minneapolis':'Midwest'},
                    2005:{'chicago':'West','albuquerque':'East','syracuse':'South','austin':'Midwest'},
                    2004:{'stlouis':'West','eastrutherford':'East','atlanta':'South','phoenix':'Midwest'},
                    2003:{'midwest':'West','west':'East','south':'South','east':'Midwest'},
                    2002:{'south':'West','west':'East','east':'South','midwest':'Midwest'}
                    }

# Round Num to Name
round_dict = {
                1:'R64',
                2:'R32',
                3:'S16',
                4:'E8',
                5:'F4'
                }

# SSL Context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Iterate years
print('Scraping Sports Reference...')
for year in years:
    print(year)
    # Sleep
    time.sleep(5)
    # Open url
    year_url = "https://www.sports-reference.com/cbb/postseason/{}-ncaa.html".format(year)
    year_url_req = Request(year_url, headers={'User-Agent': 'Mozilla/5.0'})
    year_html = urlopen(year_url_req, context=ssl_context).read()
    # Parse html
    year_soup = BeautifulSoup(year_html, features='html.parser')

    # Initialize Year
    bracket_data[year] = {}
    bracket_data[year]['NCG'] = []
    
    # Final 4
    f4_region = year_soup.select_one("div#{}".format('national'))
    # Go through all rounds and grab team name
    bracket = f4_region.find(id='bracket')
    rounds = bracket.find_all(class_="round")
    for idx, round in enumerate(rounds, start=1):
        if idx == 1:
            continue
        elif idx == 2:
            teams = round.find_all("a", href=lambda href: href and "schools" in href)
            team_names = [urlparse(team['href']).path.split('/')[3] for team in teams]
            team_names = [re.sub(r'-'," ", team.title()) for team in team_names]
            bracket_data[year]['NCG'] = team_names
        else:
            teams = round.find_all("a", href=lambda href: href and "schools" in href)
            team_names = [urlparse(team['href']).path.split('/')[3] for team in teams]
            team_names = [re.sub(r'-'," ", team.title()) for team in team_names]
            # 2024 Winner missing for some reason
            if year == 2024:
                bracket_data[year]['Winner'] = 'Connecticut'
            else:
                bracket_data[year]['Winner'] = team_names[0]

    # Try all region names
    for region in regions:
        # Get div for region
        tourney_region = year_soup.select_one("div#{}".format(region))

        # If region exists
        if tourney_region != None:      
            # go through all rounds and grab team name
            bracket_data[year][regions_convert[year][region]] = {}
            bracket = tourney_region.find(id='bracket')
            rounds = bracket.find_all(class_="round")
            for idx, round in enumerate(rounds, start=1):                
                teams = round.find_all("a", href=lambda href: href and "schools" in href)
                team_names = [urlparse(team['href']).path.split('/')[3] for team in teams]
                team_names = [re.sub(r'-'," ", team.title()) for team in team_names]
                bracket_data[year][regions_convert[year][region]][round_dict[idx]] = team_names

            # Only select 16 teams (exclude play-in)
            t = 0
            while t <= 16:
                tourney_team = tourney_region.select_one("div.round")
                t = t + 1
            else:
                playin = tourney_region.select_one('p')

                # Get each teams individual page
                for link in tourney_team.find_all('a'):
                    links = link.get('href')
                    if links.startswith('/cbb/s') == True:
                        # Sleep
                        time.sleep(5)
                        # Open url, get data
                        team_url = 'https://www.sports-reference.com' + links
                        team_html = urlopen(team_url, context=ssl_context)
                        team_soup = BeautifulSoup(team_html, features='html.parser')

                        # Create a dictionary to store scraped data
                        yeardict = {}
                        yeardict[year] = {}  

                        # Store team name
                        part_url = re.sub('https://www.sports-reference.com/cbb/schools/','',team_url)
                        yeardict[year][re.sub('/(\d+).html','',part_url).title()] = {}

                        # Homepage Texet
                        homepage = team_soup.select_one('div#info')
                        homepage_text= [hp.getText()for hp in homepage.findAll('p')]

                        # Conference
                        yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Conf'] = re.search(r'\bin\s+(.*?)\s+MBB', homepage_text[2]).group(1)

                        # Determine which round each team made it to
                        if any('Won National Final' in text for text in homepage_text):
                            yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Round'] = 7
                        elif any('Lost National Final' in text for text in homepage_text):
                            yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Round'] = 6 
                        elif any('National Semifinal' in text for text in homepage_text):
                            yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Round'] = 5
                        elif any('Regional Final' in text for text in homepage_text):
                            yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Round'] = 4
                        elif any('Regional Semifinal' in text for text in homepage_text):
                            yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Round'] = 3

                        # 2015-2011 tournaments use 2nd round as 1st & 3rd as 2nd (play-in treated as 1st round)
                        elif year in range(2011,2016):
                            if any('Third Round' in text for text in homepage_text):
                                yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Round'] = 2
                            else:
                                yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Round'] = 1
                        else:
                            if any('Second Round' in text for text in homepage_text):
                                yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Round'] = 2
                            else:
                                yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Round'] = 1
                        
                        # Adjust Wins by Rounds in Tournament
                        yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Wins'] = int(homepage_text[2][9:11]) - (yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Round']-1)
                        
                        # Determine if team won conference tourney
                        import_text = [im.getText()for im in homepage.findAll('a')]
                        if any('Tourney Champ' in text for text in import_text):
                            yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Conf Tourney'] = 1
                        else:
                            yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Conf Tourney'] = 0

                        # Add Region in Tourney
                        yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Region'] = regions_convert[year][region]

                        # Determine each team's seed in the tournamen
                        homepage_text= homepage_text[-1]
                        seeds = re.search('(\d+) seed', homepage_text)
                        # Weird instance where team forfeited
                        if (year == 2021) & (re.sub('/(\d+).html','',part_url).title() == 'Virginia-Commonwealth/Men'):
                            seeds = '10'
                        if seeds is not None:
                            if (year == 2021) & (re.sub('/(\d+).html','',part_url).title() == 'Virginia-Commonwealth/Men'):
                                yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Seed'] = '10'
                            else:
                                yeardict[year][re.sub('/(\d+).html','',part_url).title()]['Seed'] = seeds.group(0).replace(' seed','')

                            # Create df from dict
                            yeardata = pd.DataFrame.from_dict({(i,j): yeardict[i][j] 
                                                    for i in yeardict.keys() 
                                                    for j in yeardict[i].keys()},
                                                   orient='index')
                            yeardata.reset_index(inplace=True)
                            yeardata.columns = ['Year','Team','Conf','Round','Wins','Conf Tourney','Region','Seed']

                            # Convert Dtypes
                            yeardata[['Round','Wins','Conf Tourney','Seed']] = yeardata[['Round','Wins','Conf Tourney','Seed']].astype(int)

                            # Standardize Naming
                            yeardata['Team'] = yeardata['Team'].str.replace('/Men','')
                            yeardata['Team'] = yeardata['Team'].str.replace('-',' ')

                            # Create a DataFrame of the full data by appending all the dataframes created in the for loop
                            seeddata = pd.concat([seeddata,yeardata])
    
    # Unit Tests
    check_year_length_df(seeddata[seeddata['Year'] == year])
    check_year_round_length_dict(bracket_data[year])

# Export Data
# Get File Path
data_path = os.path.join(os.path.abspath(os.getcwd()), 'data/raw/sportsreference.csv')
bracket_path = os.path.join(os.path.abspath(os.getcwd()), 'data/raw/results.json')
# Export DF
seeddata.to_csv(data_path,index=False)
# Export Dictionary
with open(bracket_path, 'w') as f:
    json.dump(bracket_data, f)