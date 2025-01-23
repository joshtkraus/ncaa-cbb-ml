# Create Data Splits
def create_splits(team_data,r,best_features=None):
   # Libraries
   import pandas as pd

   # Create Response
   team_data['Outcome'] = 0
   team_data.loc[team_data['Round']<r,'Outcome'] = 0
   team_data.loc[team_data['Round']>=r,'Outcome'] = 1

   # Drop Cols
   data_sub = team_data.drop(columns=['Team','Round'])

   # Dummy Vars
   # Conference
   data_sub = pd.concat([data_sub, pd.get_dummies(data_sub['Conf'], prefix='Conf')], axis=1)
   data_sub.drop(columns='Conf', inplace=True)
   # Region
   data_sub = pd.concat([data_sub, pd.get_dummies(data_sub['Region'], prefix='Region')], axis=1)
   data_sub.drop(columns='Region', inplace=True)

   # Drop Cols from Feature Selection
   if best_features != None:
      data_sub = data_sub[best_features+['Outcome']]

   # Testing & Training Splits
   X = data_sub.drop(columns='Outcome')
   y = data_sub['Outcome']
   return X, y