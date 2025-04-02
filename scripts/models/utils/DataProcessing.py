# Create Data Splits
def create_splits(data,r,train,years_list=False,get_features=False):
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

   if get_features == True:
      return list(X.columns)
   else:
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

      # to Numpy
      X = np.array(X)    
      y = np.array(y) 

      if years_list:
         years = np.array(years)
         return X, y, years
      else:
         return X, y