# Make Picks

# Round 32
def R32(picks_dict,rg,sd,tm):
    if str(sd) in ['1','16']:
        picks_dict[rg]['R32']['1'].append(tm)
    elif str(sd) in ['8','9']:
        picks_dict[rg]['R32']['8'].append(tm)
    elif str(sd) in ['5','12']:
        picks_dict[rg]['R32']['5'].append(tm)
    elif str(sd) in ['4','13']:
        picks_dict[rg]['R32']['4'].append(tm)
    elif str(sd) in ['6','11']:
        picks_dict[rg]['R32']['6'].append(tm)
    elif str(sd) in ['3','14']:
        picks_dict[rg]['R32']['3'].append(tm)
    elif str(sd) in ['7','10']:
        picks_dict[rg]['R32']['7'].append(tm)
    else:
        picks_dict[rg]['R32']['2'].append(tm)
    return picks_dict

# Sweet 16
def S16(picks_dict,rg,sd,tm):
    if str(sd) in ['1','16','8','9']:
        picks_dict[rg]['S16']['1'].append(tm)
    elif str(sd) in ['5','12','4','13']:
        picks_dict[rg]['S16']['4'].append(tm)
    elif str(sd) in ['3','14','6','11']:
        picks_dict[rg]['S16']['3'].append(tm)
    else:
        picks_dict[rg]['S16']['2'].append(tm)
    return picks_dict

# Elite 8
def E8(picks_dict,rg,sd,tm):
    if str(sd) in ['1','16','8','9','5','12','4','13']:
        picks_dict[rg]['E8']['Upper'].append(tm)
    else:
        picks_dict[rg]['E8']['Lower'].append(tm)
    return picks_dict

# Create Picks
def create_picks(team_data):
    # Libraries
    import pandas as pd

    # Initialize
    picks_dict = {'West':{'F4':[],
                    'E8':{'Upper':[],'Lower':[]},
                    'S16':{'1':[],'4':[],'3':[],'2':[]},
                        'R32':{'1':[],'8':[],'5':[],'4':[],
                            '6':[],'3':[],'7':[],'2':[]}},
                'East':{'F4':[],
                    'E8':{'Upper':[],'Lower':[]},
                    'S16':{'1':[],'4':[],'3':[],'2':[]},
                        'R32':{'1':[],'8':[],'5':[],'4':[],
                            '6':[],'3':[],'7':[],'2':[]}},
                'South':{'F4':[],
                    'E8':{'Upper':[],'Lower':[]},
                    'S16':{'1':[],'4':[],'3':[],'2':[]},
                        'R32':{'1':[],'8':[],'5':[],'4':[],
                            '6':[],'3':[],'7':[],'2':[]}},
                'Midwest':{'F4':[],
                    'E8':{'Upper':[],'Lower':[]},
                    'S16':{'1':[],'4':[],'3':[],'2':[]},
                        'R32':{'1':[],'8':[],'5':[],'4':[],
                            '6':[],'3':[],'7':[],'2':[]}},
                'NCG':[],
            'Winner':[]
            }
    f4_cnt = 0
    e8_cnt = 0
    s16_cnt = 0
    r32_cnt = 0

    # Iterate Rounds
    for r in ['Winner','NCG','F4','E8','S16','R32']:
        # Subset data
        team_data_sub = team_data[['Team','Seed','Region',r]]
        pred = team_data_sub.sort_values(by=r, ascending=False)
        pred.reset_index(inplace=True)
        
        # Winner
        if r == 'Winner':
            # team identifiers
            tm = pred['Team'][0]
            sd = team_data.loc[team_data['Team']==tm,'Seed'].item()
            rg = team_data.loc[team_data['Team']==tm,'Region'].item()
            win_reg = rg
            # Winner
            picks_dict['Winner'].append(tm)
            # NCG
            picks_dict['NCG'].append(tm)
            # F4
            picks_dict[rg]['F4'] = tm
            # E8
            picks_dict = E8(picks_dict,rg,sd,tm)
            # S16
            picks_dict = S16(picks_dict,rg,sd,tm)
            # R32
            picks_dict = R32(picks_dict,rg,sd,tm)
            # increase counter
            f4_cnt += 1
            e8_cnt += 1
            s16_cnt += 1
            r32_cnt += 1
        # NCG
        elif r == 'NCG':
            i = 0
            while len(picks_dict['NCG']) < 2:
                # team identifiers
                tm = pred['Team'][i]
                sd = team_data.loc[team_data['Team']==tm,'Seed'].item()
                rg = team_data.loc[team_data['Team']==tm,'Region'].item()
                if (((win_reg == 'West')|(win_reg == 'East')) & ((rg != 'West')&(rg != 'East')))|(((win_reg == 'South')|(win_reg == 'Midwest')) & ((rg != 'South')&(rg != 'Midwest'))):
                    # add to round
                    picks_dict['NCG'].append(tm)
                    # F4
                    picks_dict[rg]['F4'] = tm
                    # E8
                    picks_dict = E8(picks_dict,rg,sd,tm)
                    # S16
                    picks_dict = S16(picks_dict,rg,sd,tm)
                    # R32
                    picks_dict = R32(picks_dict,rg,sd,tm)
                    # increase counter
                    f4_cnt += 1
                    e8_cnt += 1
                    s16_cnt += 1
                    r32_cnt += 1
                i += 1
        # F4
        elif r == 'F4':
            i = 0
            while f4_cnt < 4:
                # team identifiers
                tm = pred['Team'][i]
                sd = team_data.loc[team_data['Team']==tm,'Seed'].item()
                rg = team_data.loc[team_data['Team']==tm,'Region'].item()
                if len(picks_dict[rg]['F4']) == 0:
                    # F4
                    picks_dict[rg]['F4'] = tm
                    # E8
                    picks_dict = E8(picks_dict,rg,sd,tm)
                    # S16
                    picks_dict = S16(picks_dict,rg,sd,tm)
                    # R32
                    picks_dict = R32(picks_dict,rg,sd,tm)
                    # increase counter
                    f4_cnt += 1
                    e8_cnt += 1
                    s16_cnt += 1
                    r32_cnt += 1
                i += 1
        # E8
        elif r == 'E8':
            i = 0
            while e8_cnt < 8:
                found = False
                # team identifiers
                tm = pred['Team'][i]
                sd = team_data.loc[team_data['Team']==tm,'Seed'].item()
                rg = team_data.loc[team_data['Team']==tm,'Region'].item()
                # E8
                if str(sd) in ['1','16','8','9','5','12','4','13']:
                    if len(picks_dict[rg]['E8']['Upper']) == 0:
                        picks_dict[rg]['E8']['Upper'].append(tm)
                        found = True
                else:
                    if len(picks_dict[rg]['E8']['Lower']) == 0:
                        picks_dict[rg]['E8']['Lower'].append(tm)
                        found = True
                if found == True:
                    # S16
                    picks_dict = S16(picks_dict,rg,sd,tm)
                    # R32
                    picks_dict = R32(picks_dict,rg,sd,tm)
                    # increase counter
                    e8_cnt += 1
                    s16_cnt += 1
                    r32_cnt += 1
                i+= 1
        # S16
        elif r == 'S16':
            i = 0
            while s16_cnt < 16:
                found = False
                # team identifiers
                tm = pred['Team'][i]
                sd = team_data.loc[team_data['Team']==tm,'Seed'].item()
                rg = team_data.loc[team_data['Team']==tm,'Region'].item()
                # S16
                if str(sd) in ['1','16','8','9']:
                    if len(picks_dict[rg]['S16']['1']) == 0:
                        picks_dict[rg]['S16']['1'].append(tm)
                        found = True
                elif str(sd) in ['5','12','4','13']:
                    if len(picks_dict[rg]['S16']['4']) == 0:
                        picks_dict[rg]['S16']['4'].append(tm)
                        found = True
                elif str(sd) in ['3','14','6','11']:
                    if len(picks_dict[rg]['S16']['3']) == 0:
                        picks_dict[rg]['S16']['3'].append(tm)
                        found = True
                else:
                    if len(picks_dict[rg]['S16']['2']) == 0:
                        picks_dict[rg]['S16']['2'].append(tm)
                        found = True
                if found == True:
                    # R32
                    picks_dict = R32(picks_dict,rg,sd,tm)
                    # increase counter
                    s16_cnt += 1
                    r32_cnt += 1
                i += 1
        # R32
        else:
            i = 0
            while r32_cnt < 32:
                # team identifiers
                tm = pred['Team'][i]
                sd = team_data.loc[team_data['Team']==tm,'Seed'].item()
                rg = team_data.loc[team_data['Team']==tm,'Region'].item()
                # R32
                if str(sd) in ['1','16']:
                    if len(picks_dict[rg]['R32']['1']) == 0:
                        picks_dict[rg]['R32']['1'].append(tm)
                        r32_cnt += 1
                elif str(sd) in ['8','9']:
                    if len(picks_dict[rg]['R32']['8']) == 0:
                        picks_dict[rg]['R32']['8'].append(tm)
                        r32_cnt += 1
                elif str(sd) in ['5','12']:
                    if len(picks_dict[rg]['R32']['5']) == 0:
                        picks_dict[rg]['R32']['5'].append(tm)
                        r32_cnt += 1
                elif str(sd) in ['4','13']:
                    if len(picks_dict[rg]['R32']['4']) == 0:
                        picks_dict[rg]['R32']['4'].append(tm)
                        r32_cnt += 1
                elif str(sd) in ['6','11']:
                    if len(picks_dict[rg]['R32']['6']) == 0:
                        picks_dict[rg]['R32']['6'].append(tm)
                        r32_cnt += 1
                elif str(sd) in ['3','14']:
                    if len(picks_dict[rg]['R32']['3']) == 0:
                        picks_dict[rg]['R32']['3'].append(tm)
                        r32_cnt += 1
                elif str(sd) in ['7','10']:
                    if len(picks_dict[rg]['R32']['7']) == 0:
                        picks_dict[rg]['R32']['7'].append(tm)
                        r32_cnt += 1
                else:
                    if len(picks_dict[rg]['R32']['2']) == 0:
                        picks_dict[rg]['R32']['2'].append(tm)
                        r32_cnt += 1
                # increase counter
                i += 1
    return picks_dict

# Calculate Real Points
def real_Bracket(picks,real):
    # Initialize
    point_totals = {'R32':10,
                    'S16':20,
                    'E8':40,
                    'F4':80,
                    'NCG':160,
                    'Winner':320}
    accuracy = {'R32':0,
                'S16':0,
                'E8':0,
                'F4':0,
                'NCG':0,
                'Winner':0}
    total = 0

    # Iterate through region, round, and picks
    for region in ['West','East','South','Midwest']:    
        for rd in ['R32','S16','E8','F4']:
            team_list = picks[region][rd]
            if rd != 'F4':
                team_list = [item for sublist in team_list.values() for item in sublist]
            else:
                team_list = [team_list]
            for team in team_list:
                # find correct picks
                if team in real[region][rd]:
                    # add points
                    total+=point_totals[rd]
                    accuracy[rd] += 1
    # F4
    for team in picks['NCG']:
        # find correct picks
        if team in real['NCG']:
            # add points
            total+=point_totals['NCG'] 
            accuracy['NCG'] += 1
    # NCG
    team = picks['Winner'][0]
    # find correct picks
    if team == real['Winner']:
        # add points
        total+=point_totals['Winner']
        accuracy['Winner'] += 1
    return total, accuracy

# Function to Make Picks & Calculate Points
def predict_bracket(team_data, real_picks=None, calc_correct=True):
    # Make Picks
    picks = create_picks(team_data)

    # Calculating Points from Actual Results
    if calc_correct == True:
        # real picks
        points, accs = real_Bracket(picks,real_picks)
        return picks, points, accs
    else:
        return picks