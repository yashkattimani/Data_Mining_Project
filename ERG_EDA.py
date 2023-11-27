#%% [markdown]

# Team 6 Project Code

# Emily's data cleaning and variable renaming for Team use

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#import sklearn
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
# %%

# --> ERG Note to team: rename csv as "acc.csv"

#dfacc = pd.read_csv('acc.csv')
# %%
dfacc = pd.read_csv('acc.csv')

# %%
dfacc.info()

#%%
# ERG Note: Original df info follows below

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 2040203 entries, 0 to 2040202
# Data columns (total 29 columns):
#   Column                         Dtype  
# ---  ------                         -----  
# 0   CRASH DATE                     object 
# 1   CRASH TIME                     object 
# 2   BOROUGH                        object 
# 3   ZIP CODE                       object 
# 4   LATITUDE                       float64
# 5   LONGITUDE                      float64
# 6   LOCATION                       object 
# 7   ON STREET NAME                 object 
# 8   CROSS STREET NAME              object 
# 9   OFF STREET NAME                object 
# 10  NUMBER OF PERSONS INJURED      float64
# 11  NUMBER OF PERSONS KILLED       float64
# 12  NUMBER OF PEDESTRIANS INJURED  int64  
# 13  NUMBER OF PEDESTRIANS KILLED   int64  
# 14  NUMBER OF CYCLIST INJURED      int64  
# 15  NUMBER OF CYCLIST KILLED       int64  
# 16  NUMBER OF MOTORIST INJURED     int64  
# 17  NUMBER OF MOTORIST KILLED      int64  
# 18  CONTRIBUTING FACTOR VEHICLE 1  object 
# 19  CONTRIBUTING FACTOR VEHICLE 2  object 
# 20  CONTRIBUTING FACTOR VEHICLE 3  object 
# 21  CONTRIBUTING FACTOR VEHICLE 4  object 
# 22  CONTRIBUTING FACTOR VEHICLE 5  object 
# 23  CONTRIBUTING FACTOR VEHICLE 2  object
# 24  COLLISION_ID                   int64
# 25  VEHICLE TYPE CODE 1            object 
# 26  VEHICLE TYPE CODE 2            object 
# 27  VEHICLE TYPE CODE 3            object 
# 27  VEHICLE TYPE CODE 4            object 
# 28  VEHICLE TYPE CODE 5            object 
# dtypes: float64(4), int64(7), object(18)
# memory usage: 451.4+ MB

# %% [markdown]
dfacc.describe().round(2)

#ERG Observations:
# Whoah... there was a crash that injured 43 people!
# ~1.8 M records have lat/long; 2.0M have injury data.

#%%

# ERG Note: Here are the original variables:
# ['CRASH DATE', 'CRASH TIME', 'BOROUGH', 'ZIP CODE', 'LATITUDE',
#       'LONGITUDE', 'LOCATION', 'ON STREET NAME', 'CROSS STREET NAME',
#       'OFF STREET NAME', 'NUMBER OF PERSONS INJURED',
#       'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS INJURED',
#       'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF CYCLIST INJURED',
#       'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST INJURED',
#       'NUMBER OF MOTORIST KILLED', 'CONTRIBUTING FACTOR VEHICLE 1',
#       'CONTRIBUTING FACTOR VEHICLE 2', 'CONTRIBUTING FACTOR VEHICLE 3',
#       'CONTRIBUTING FACTOR VEHICLE 4', 'CONTRIBUTING FACTOR VEHICLE 5',
#       'COLLISION_ID', 'VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2',
#       'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5'] 

# ERG Note: Here is the mapping of old 

# 'CRASH DATE'--> 'DATE'
# 'CRASH TIME' --> 'TIME'
# 'BOROUGH' --> , 'BOROUGH'
# 'ZIP CODE' --> , 'ZIP'
# 'LATITUDE' --> , 'LAT',
# 'LONGITUDE' --> 'LONG', 
# 'LOCATION' --> 'LOC'
# 'ON STREET NAME'--> 'ONST'
# 'CROSS STREET NAME' --> 'CRST'
# 'OFF STREET NAME' --> 'OFFST'
# 'NUMBER OF PERSONS INJURED' --> 'NUMINJ'
# 'NUMBER OF PERSONS KILLED' --> 'NUMKIL'
# 'NUMBER OF PEDESTRIANS INJURED' --> 'NUMPEDINJ'
# 'NUMBER OF PEDESTRIANS KILLED' --> 'NUMPEDKIL'
# 'NUMBER OF CYCLIST INJURED' --> 'NUMCYCINJ'
# 'NUMBER OF CYCLIST KILLED' --> 'NUMCYCKIL'
# 'NUMBER OF MOTORIST INJURED' --> 'NUMMOTINJ'
# 'NUMBER OF MOTORIST KILLED' --> 'NUMMOTKIL' 
# 'CONTRIBUTING FACTOR VEHICLE 1' --> 'CFV1'
# 'CONTRIBUTING FACTOR VEHICLE 2' --> 'CFV2'
# 'CONTRIBUTING FACTOR VEHICLE 3' --> 'CFV3'
# 'CONTRIBUTING FACTOR VEHICLE 4' --> 'CFV4'
# 'CONTRIBUTING FACTOR VEHICLE 5' --> 'CFV5'
# 'COLLISION_ID' --> 'ID'
# 'VEHICLE TYPE CODE 1' --> 'V1'
# 'VEHICLE TYPE CODE 2' --> 'V2'
# 'VEHICLE TYPE CODE 3' --> 'V3'
# 'VEHICLE TYPE CODE 4' --> 'V4'
# 'VEHICLE TYPE CODE 5' --> 'V5'

#%%
# Changing columns names
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
# Now, correcting data types

#%%
# ERG Note: designating categoricals as such
dfacc[['BOROUGH', 'LOC', 'ZIP','ONST', 'CRST', 'CFV1',
       'CFV2', 'CFV3',
       'CFV4', 'CFV5', 'V1', 'V2',
       'V3', 'V4', 'V5']] = dfacc[['BOROUGH', 'LOC', 'ZIP','ONST', 'CRST', 'CFV1',
       'CFV2', 'CFV3',
       'CFV4', 'CFV5', 'V1', 'V2',
       'V3', 'V4', 'V5']].astype("category")

# ERG Note: Date and time
#%%
# ERG Note: creating new column combining date and time as pandas datetime type
dfacc["DATETIME"] = pd.to_datetime(dfacc["DATE"] + " " + dfacc["TIME"])
pd.to_datetime(dfacc["DATETIME"])
#pd.to_datetime(dfacc['TIME'])
#dfacc[["DATE", "TIME"]]=dfacc[["DATE", "TIME"]].astype("datetime64[ns]")
#dfacc['NUMINJ', 'NUMKIL', 'NUMPEDINJ', 'NUMPEDKIL', 'NUMCYCINJ', 'NUMCYCKIL', 'NUMMOTINJ', 'NUMMOTKIL'] = dfacc['NUMINJ', 'NUMKIL', 'NUMPEDINJ', 'NUMPEDKIL', 'NUMCYCINJ', 'NUMCYCKIL', 'NUMMOTINJ', 'NUMMOTKIL'].astype("int")

#%%
# ERG Note: Adding separate columns for year and month

dfacc["YEAR"] = dfacc["DATETIME"].dt.year
dfacc["MONTH"] = dfacc["DATETIME"].dt.month
dfacc["DAY"] = dfacc["DATETIME"].dt.day

# %%
# add a column for month and year
dfacc['monyear'] = dfacc['DATETIME'].dt.to_period('M')
# add a column with value one to permit counting of accidents later.
dfacc["COUNT"] = 1
#dfacc_monyear_harms = pd.DataFrame(dfacc[["monyear", "NUMINJ", "NUMKIL"]])

# add columns for whether there were injuries and whether their were deaths
dfacc["INJBOL"] = dfacc["NUMINJ"] > 0
dfacc["KILBOL"] = dfacc["NUMKIL"] > 0

#%%
print(dfacc.info())

#%%
sns.histplot(data = dfacc, hue = "BOROUGH", x = "NUMKIL", multiple = "stack")
# Most accidents don't lead to deaths. No accident caused more than 8 deaths. (Assuming the records are complete + accurate.)

#%%
# average number of deathsper accident
sns.barplot(data = dfacc, hue = "BOROUGH", x = "BOROUGH", y = "NUMKIL", legend = False)

#%%
sns.histplot(data = dfacc, hue = "BOROUGH", x = "NUMINJ", multiple = "stack")
# 
#%%
# average number of injuries per accident
sns.barplot(data = dfacc, hue = "BOROUGH", x = "BOROUGH", y = "NUMINJ", legend = False)
# 

#%%
sns.histplot(data = dfacc, x = "BOROUGH", hue = "BOROUGH", legend = False)
# Over the full time window, Brooklyn has had the most accidents, 
# followed by Queens, Manhattan, the Bronx, and Statten Island.

#%%
sns.histplot(data = dfacc, bins = 136, hue = "BOROUGH", x = "DATETIME", multiple = "stack")
# There is a clear drop in # accidents after pandemic onset!
# There does appear to be a cyclical variation in the number of accidents!
#%%
#sns.histplot(data = dfacc, hue = "BOROUGH", x = "monyear")

#%%
# Let's look by month: 
sns.histplot(data = dfacc, x = "MONTH")
# Seems to be... we should check whether this is statistically significant.
# But I wonder if the transition to COVID Lockdown has an influence here? 

#%%
# Let's look before COVID lockdown (roughly defined as before 2020)
dfacc_bef = dfacc.loc[(dfacc["YEAR"] < 2020)]
sns.histplot(data = dfacc_bef, hue = "BOROUGH", x = "MONTH", multiple = "stack")

# Seems a bit more pronounced. Worth testing further... later!

#%%
# Let's look after COVID lockdown (after 2020)
dfacc_aft = dfacc.loc[(dfacc["YEAR"] > 2020)]
sns.histplot(data = dfacc_aft, hue = "BOROUGH", x = "MONTH")

#%%

# Now for some temporal trends!
sns.scatterplot(x = "DATETIME", y = "NUMINJ",
             hue = "BOROUGH", data = dfacc)

#%%
sns.scatterplot(x = "DATETIME", y = "NUMKIL",
             hue = "BOROUGH", data = dfacc)

#%%
import statsmodels.api as sm
from statsmodels.formula.api import glm
modelINJ = glm ( formula = 'INJBOL ~ MONTH + YEAR + BOROUGH + CFV1', data = dfacc , family = sm.families.Binomial())

modelINJ = modelINJ.fit()
print( modelINJ.summary() )

#%%
#dacc_mytest = dfacc.groupby(["monyear", "BOROUGH"]).NUMKIL.sum().reset_index()
#%%
#dacc_mytest_inj = dfacc.groupby(["monyear", "BOROUGH"]).NUMINJ.sum().reset_index()

#%%
#dacc_mytest_inj["monyear"] = dacc_mytest_inj["monyear"].astype("datetime64[ns]")
#sns.scatterplot(x = "monyear", y = "NUMINJ",
#             hue = "BOROUGH", data = dacc_mytest_inj)

#%%
#dacc_mytest["monyear"] = dacc_mytest["monyear"].astype("datetime64[ns]")
#sns.scatterplot(x = "monyear", y = "NUMKIL", hue = "BOROUGH", data = dacc_mytest )

#%%
# New efforts 11/25/2023

#dfacc["INJBOL"] = dfacc["NUMINJ"] > 0
#dfacc["KILBOL"] = dfacc["NUMKIL"] > 0

#%%
# Adding Creating monthly version of original dfacc
dfacc_monthly = dfacc.groupby(["monyear", "BOROUGH"]).COUNT.sum().reset_index()

# creating monthly for each of the elements of interest by borough 
dfacc_monthly_inj = dfacc.groupby(["monyear", "BOROUGH"]).NUMINJ.sum().reset_index()
dfacc_monthly_injbol = dfacc.groupby(["monyear", "BOROUGH"]).INJBOL.sum().reset_index()
dfacc_monthly_kil = dfacc.groupby(["monyear", "BOROUGH"]).NUMKIL.sum().reset_index()
dfacc_monthly_kilbol = dfacc.groupby(["monyear", "BOROUGH"]).KILBOL.sum().reset_index()

#%%
# And now adding these columns to the monthly df

dfacc_monthly["NUMINJ"] = dfacc_monthly_inj["NUMINJ"]
dfacc_monthly["NUMACCINJ"] = dfacc_monthly_injbol["INJBOL"]
dfacc_monthly["NUMKIL"] = dfacc_monthly_kil["NUMKIL"]
dfacc_monthly["NUMACCKIL"] = dfacc_monthly_kilbol["KILBOL"]

#%%
dfacc_monthly["SHAREACCINJ"] = dfacc_monthly["NUMACCINJ"] / dfacc_monthly["COUNT"]
dfacc_monthly["SHAREACCKIL"] = dfacc_monthly["NUMACCKIL"] / dfacc_monthly["COUNT"]  

#%%
# Converting back to datetime
dfacc_monthly["monyear"] = dfacc_monthly["monyear"].astype("datetime64[ns]")

#%%
# Plotting share of accidents each month that yield injury.
sns.scatterplot(data = dfacc_monthly, hue = "BOROUGH", x = "monyear", y = "NUMACCINJ", legend = False)
#%%
# Plotting share of accidents each month that yield injury.
sns.scatterplot(data = dfacc_monthly, hue = "BOROUGH", x = "monyear", y = "SHAREACCINJ")
#%%
# Plotting share of accidents each month that yield injury.
sns.scatterplot(data = dfacc_monthly, hue = "BOROUGH", x = "monyear", y = "SHAREACCKIL")

#%%
dfacc_monthly["YEAR"] = dfacc_monthly["monyear"].dt.year
dfacc_monthly["MONTH"] = dfacc_monthly["monyear"].dt.month

dfacc_monthly["YEAR"] = dfacc_monthly["YEAR"].astype("category")
dfacc_monthly["MONTH"] = dfacc_monthly["MONTH"].astype("category")

#%%
# Now trying a T-test
# Null hypothesis, H_0: The average number of accidents report per month is the same before and after the onset of the pandemic.
# Alternate hypothesis, H_a: There is a statistically significant difference between the average number of accidents reported before and after the
# pandemic began.

import scipy
scipy.stats.ttest_ind(dfacc_monthly["COUNT"])

#%%
# (data = dfacc_monthly, x = "monyear", y = "SHAREACCINJ")

#%%
# Subsetting df to pre- and post- lockdown date.
# Lockdown date determined by date cited in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8806179/
#dfacc_pre = dfacc.loc[(dfacc["DATETIME"] < "2020-03-23")]
#dfacc_post = dfacc.loc[(dfacc["DATETIME"] >= "2020-03-23")]

#%%
# Converting to monthly totals
dfacc_pre_g = dfacc_pre.groupby(["monyear", "BOROUGH"]).NUMINJ.sum().reset_index()
dfacc_post_g = dfacc_post.groupby(["monyear", "BOROUGH"]).NUMINJ.sum().reset_index()


#%%
#dacc_pre_g["monyear"] = dacc_pre_g["monyear"].astype("datetime64[ns]")
#dacc_post_g["monyear"] = dacc_post_g["monyear"].astype("datetime64[ns]")
#Now, to find the average number of accidents per month pre- and post-lockdown.

#%%


#%%
dfacc_pre_g_tot = dfacc_pre.groupby(["monyear", "BOROUGH"]).COUNT.count().reset_index()
dfacc_post_g_tot = dfacc_post.groupby(["monyear", "BOROUGH"]).COUNT.count().reset_index()

#%%
dfacc_pre_g["TOT"] = dfacc_pre_g_tot["COUNT"]
dfacc_pre_g["SHAREINJ"] = dfacc_pre_g["NUMINJ"] / dfacc_pre_g["TOT"] 
dfacc_post_g["TOT"] = dfacc_post_g_tot["COUNT"]
dfacc_post_g["SHAREINJ"] = dfacc_post_g["NUMINJ"] / dfacc_post_g["TOT"]


#%%

# Converting back to datetime
dfacc_pre_g["monyear"] = dfacc_pre_g["monyear"].astype("datetime64[ns]")
dfacc_post_g["monyear"] = dfacc_post_g["monyear"].astype("datetime64[ns]")

#%%
sns.histplot(data = dfacc_post, hue = "BOROUGH", x = "monyear")

#%%
# Plotting share of accidents each month that yield injury.
sns.scatterplot(x = "monyear", y = "SHAREINJ",
             hue = "BOROUGH", data = dfacc_pre_g)

#%%
# Plotting share of accidents each month that yield injury.
sns.scatterplot(x = "monyear", y = "SHAREINJ",
             hue = "BOROUGH", data = dfacc_post_g)

#%%

#%%

sns.scatterplot(x = "monyear", y = "NUMINJ",
             hue = "BOROUGH", data = dfacc)

#%%
### Old cleaning attempts follow below.
#%%
#Experimenting with fixing V1!!

#dfacc["V1"].astype("string")
#dfacc["V1"] = dfacc["V1"].str.lower()
#sns.histplot(data = dfacc, x = "V1")
dfacc_V1cts = dfacc.value_counts("V1")
dfacc_V1cts = dfacc_V1cts.to_frame()
dfacc_V1cts.loc[(dfacc_V1cts["count"] > 1000)]
dfacc_V1cts1000 = dfacc_V1cts.loc[(dfacc_V1cts["count"] > 1000)]
print(dfacc_V1cts1000)
#Here are all V1 types that were the primary vehicle in at least 1000 accidents

#                                      count
# V1                                         
# Sedan                                560517
# Station Wagon/Sport Utility Vehicle  441120
# PASSENGER VEHICLE                    416206
# SPORT UTILITY / STATION WAGON        180291
# Taxi                                  50540
# 4 dr sedan                            40164
# Pick-up Truck                         33860
# TAXI                                  31911
# VAN                                   25266
# Box Truck                             23789
# OTHER                                 22968
# Bus                                   20808
# UNKNOWN                               19937
# Bike                                  14404
# LARGE COM VEH(6 OR MORE TIRES)        14397
# BUS                                   13993
# SMALL COM VEH(4 TIRES)                13216
# PICK-UP TRUCK                         11505
# LIVERY VEHICLE                        10481
# Tractor Truck Diesel                  10121
# Van                                    8788
# Motorcycle                             7895
# Ambulance                              4255
# MOTORCYCLE                             4195
# Convertible                            3675
# Dump                                   3674
# E-Bike                                 2920
# 2 dr sedan                             2653
# AMBULANCE                              2615
# PK                                     2418
# Flat Bed                               2347
# Garbage or Refuse                      2186
# E-Scooter                              1969
# Carry All                              1867
# Tractor Truck Gasoline                 1534
# Moped                                  1519
# Tow Truck / Wrecker                    1304


# ["Sedan", "Station Wagon/Sport Utility Vehicle", "PASSENGER VEHICLE", "SPORT UTILITY / STATION WAGON", "Taxi", "4 dr sedan", "Pick-up Truck", "TAXI", "VAN", "Box Truck", "OTHER", "Bus", "UNKNOWN", "Bike", "LARGE COM VEH(6 OR MORE TIRES)", "BUS", "SMALL COM VEH(4 TIRES)", "PICK-UP TRUCK", "LIVERY VEHICLE", "Tractor Truck Diesel","Van", "Motorcycle", "Ambulance", "MOTORCYCLE", "Convertible", "Dump", "E-Bike", "2 dr sedan", "AMBULANCE", "PK", "Flat Bed", "Garbage or Refuse", "E-Scooter", "Carry All", "Tractor Truck Gasoline", "Moped", "Tow Truck / Wrecker"]
#%%

#def V1clean(row):
# define the rarer V1 values as "Other2" to make analysis more tractable
#    for i in row["V1"]:
#        if i in ["Sedan", "Station Wagon/Sport Utility Vehicle", "PASSENGER VEHICLE", "SPORT UTILITY / STATION WAGON", "Taxi", "4 dr sedan", "Pick-up Truck", "TAXI", "VAN", "Box Truck", "OTHER", "Bus", "UNKNOWN", "Bike", "LARGE COM VEH(6 OR MORE TIRES)", "BUS", "SMALL COM VEH(4 TIRES)", "PICK-UP TRUCK", "LIVERY VEHICLE", "Tractor Truck Diesel","Van", "Motorcycle", "Ambulance", "MOTORCYCLE", "Convertible", "Dump", "E-Bike", "2 dr sedan", "AMBULANCE", "PK", "Flat Bed", "Garbage or Refuse", "E-Scooter", "Carry All", "Tractor Truck Gasoline", "Moped", "Tow Truck / Wrecker"]:
#            pass
#        else: dfacc[i] == "Other2"
#    return

#%%

for i in range(0, 2040200):
    if dfacc["V1"][i] not in ["Sedan", "Station Wagon/Sport Utility Vehicle", "PASSENGER VEHICLE", "SPORT UTILITY / STATION WAGON", "Taxi", "4 dr sedan", "Pick-up Truck", "TAXI", "VAN", "Box Truck", "OTHER", "Bus", "UNKNOWN", "Bike", "LARGE COM VEH(6 OR MORE TIRES)", "BUS", "SMALL COM VEH(4 TIRES)", "PICK-UP TRUCK", "LIVERY VEHICLE", "Tractor Truck Diesel","Van", "Motorcycle", "Ambulance", "MOTORCYCLE", "Convertible", "Dump", "E-Bike", "2 dr sedan", "AMBULANCE", "PK", "Flat Bed", "Garbage or Refuse", "E-Scooter", "Carry All", "Tractor Truck Gasoline", "Moped", "Tow Truck / Wrecker"]:
        dfacc["V1"][i] == "Other2"

#%%
dfacc['V1'] = dfacc.apply(V1clean, axis=1)

#%%
dfacc_V1cl = dfacc.loc[(dfacc["V1"] in ["Sedan", "Station Wagon/Sport Utility Vehicle", "PASSENGER VEHICLE", "SPORT UTILITY / STATION WAGON", "Taxi", "4 dr sedan", "Pick-up Truck", "TAXI", "VAN", "Box Truck", "OTHER", "Bus", "UNKNOWN", "Bike", "LARGE COM VEH(6 OR MORE TIRES)", "BUS", "SMALL COM VEH(4 TIRES)", "PICK-UP TRUCK", "LIVERY VEHICLE", "Tractor Truck Diesel","Van", "Motorcycle", "Ambulance", "MOTORCYCLE", "Convertible", "Dump", "E-Bike", "2 dr sedan", "AMBULANCE", "PK", "Flat Bed", "Garbage or Refuse", "E-Scooter", "Carry All", "Tractor Truck Gasoline", "Moped", "Tow Truck / Wrecker"])]
#%%
#dfacc_V1 = 
#print(dfacc.value_counts("V1"))

# %%

#sns.pairplot(dfacc, hue = "V1")
# %%
# How many people died total in the records?

#print ( dfacc["NUMKIL"].sum() )

# How many people died total in the records?

#for dfacc["DATE"]."year"]in 2019 v. 2021

# %%
