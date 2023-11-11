#%% [markdown]

# Team 6 Project Code

import pandas as pd
import rfit 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
# %%
#dfacc = pd.read_csv('acc.csv')
# %%

dfacc = pd.read_csv('acc.csv')

#%% [markdown]

# Emily's EDA
# %%
dfacc.info()
# %%
dfacc.describe().round(2)
# %% [markdown]

# Whoah... there was a crash that injured 43 people!
# ~1.8 M records have lat/long; 2.0M have injury data.

#%%
# Changing columns names
#['CRASH DATE', 'CRASH TIME', 'BOROUGH', 'ZIP CODE', 'LATITUDE',
#       'LONGITUDE', 'LOCATION', 'ON STREET NAME', 'CROSS STREET NAME',
#       'OFF STREET NAME', 'NUMBER OF PERSONS INJURED',
#       'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS INJURED',
#       'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF CYCLIST INJURED',
#       'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST INJURED',
#       'NUMBER OF MOTORIST KILLED', 'CONTRIBUTING FACTOR VEHICLE 1',
#       'CONTRIBUTING FACTOR VEHICLE 2', 'CONTRIBUTING FACTOR VEHICLE 3',
#       'CONTRIBUTING FACTOR VEHICLE 4', 'CONTRIBUTING FACTOR VEHICLE 5',
#       'COLLISION_ID', 'VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2',
#       'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5'] = 

dfacc.columns = ['DATE', 'TIME', 'BOROUGH', 'ZIP', 'LAT',
       'LONG', 'LOC', 'ONST', 'CRST',
       'OFFST', 'NUMINJ',
       'NUMKIL', 'NUMPEDINJ',
       'NUMPEDKIL', 'NUMCYCINJ',
       'NUMCYCKIL', 'NUMMOTINJ',
       'NUMMOTKIL', 'CFV1',
       'CFV2', 'CFV3',
       'CFV4', 'CFV5',
       'ID', 'V1', 'V2',
       'V3', 'V4', 'V5']


#%%
dfacc.info()
# %%
dfacc.describe().round(2)

#%%
# Correcting data types

dfacc[['BOROUGH', 'LOC', 'ZIP','ONST', 'CRST', 'CFV1',
       'CFV2', 'CFV3',
       'CFV4', 'CFV5', 'V1', 'V2',
       'V3', 'V4', 'V5']]=dfacc[['BOROUGH', 'LOC', 'ZIP','ONST', 'CRST', 'CFV1',
       'CFV2', 'CFV3',
       'CFV4', 'CFV5', 'V1', 'V2',
       'V3', 'V4', 'V5']].astype("category")
dfacc["DATE"]=dfacc["DATE"].astype("datetime64[ns]")
#dfacc['NUMINJ', 'NUMKIL', 'NUMPEDINJ', 'NUMPEDKIL', 'NUMCYCINJ', 'NUMCYCKIL', 'NUMMOTINJ', 'NUMMOTKIL'] = dfacc['NUMINJ', 'NUMKIL', 'NUMPEDINJ', 'NUMPEDKIL', 'NUMCYCINJ', 'NUMCYCKIL', 'NUMMOTINJ', 'NUMMOTKIL'].astype("int")

dfacc["TIME"]=dfacc["TIME"].astype("datetime64[ns]")

#%%

print(dfacc.info())

#%%
sns.histplot(data = dfacc, x = "NUMKIL")
# Most accidents don't lead to deaths. No accident cause more than 8 deaths.

#%%
sns.histplot(data = dfacc, x = "NUMINJ")
#%%
sns.histplot(data = dfacc, x = "BOROUGH")

#%%
sns.histplot(data = dfacc, x = "DATE")

#%%
#sns.histplot(data = dfacc, x = "TIME")

# %%

#sns.pairplot(dfacc, hue = "V1")
# %%
# How many people died total in the records?

print ( dfacc["NUMKIL"].sum() )

# How many people died total in the records?

for dfacc["DATE"]."year"]in 2019 v. 2021

# %%
