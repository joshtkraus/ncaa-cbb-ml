def get_modeling_cols(data):
   # Libraries
   import pandas as pd

   # Drop Cols
   data_sub = data.drop(columns=['Team','Round'])

   # Dummy Vars
   # Conference
   data_sub = pd.concat([data_sub, pd.get_dummies(data_sub['Conf'], prefix='Conf')], axis=1)
   data_sub.drop(columns='Conf', inplace=True)
   # Region
   data_sub = pd.concat([data_sub, pd.get_dummies(data_sub['Region'], prefix='Region')], axis=1)
   data_sub.drop(columns='Region', inplace=True)

   return list(data_sub.columns)

# Create Data Splits
def create_splits(data,r,train,best_features=None,years_list=False):
   # Libraries
   import numpy as np
   import pandas as pd
   from sklearn.preprocessing import MinMaxScaler
   from imblearn.over_sampling import BorderlineSMOTE
   from imblearn.under_sampling import TomekLinks

   # Create Response
   mod_data = data.copy()
   mod_data['Outcome'] = 0
   mod_data.loc[mod_data['Round']<r,'Outcome'] = 0
   mod_data.loc[mod_data['Round']>=r,'Outcome'] = 1

   # Drop Cols
   data_sub = mod_data.drop(columns=['Team','Round'])

   # Dummy Vars
   # Conference
   data_sub = pd.concat([data_sub, pd.get_dummies(data_sub['Conf'], prefix='Conf')], axis=1)
   data_sub.drop(columns='Conf', inplace=True)
   # Region
   data_sub = pd.concat([data_sub, pd.get_dummies(data_sub['Region'], prefix='Region')], axis=1)
   data_sub.drop(columns='Region', inplace=True)

   # Testing & Training Splits
   X = data_sub.drop(columns='Outcome')
   y = data_sub['Outcome']

   # If Best Features, get col indicies
   if best_features != None:
      if isinstance(best_features,dict):
         col_ind = X.columns.get_indexer(best_features[r])
      else:
         col_ind = X.columns.get_indexer(best_features)

   # Data Processing
   # Scaling
   scaler = MinMaxScaler()
   X = scaler.fit_transform(X)

   if train:
      # SMOTE
      sm = BorderlineSMOTE(random_state=23)
      X, y = sm.fit_resample(X, y)

      # Tomek Links
      tl = TomekLinks()
      X, y = tl.fit_resample(X, y)

   # Get Years
   if years_list:
      years = X[:,0]

   # If Best Features, subset cols
   if best_features is not None:
      X = X[:,col_ind]

   # to Numpy
   X = np.array(X)    
   y = np.array(y) 

   if years_list:
      years = np.array(years)
      return X, y, years
   else:
      return X, y