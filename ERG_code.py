#%% [markdown]

# Team 6 Project Code

# Emily's data cleaning and variable renaming for Team use

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
import sklearn
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
# looking at NAs by variable
dfacc.isna().sum(axis = 0, numeric_only=False)

#%%
# # looking at total # NAs
dfacc.isna().sum(axis = 0, numeric_only=False).sum()



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
# ERG Note Changing columns names
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
       'V3', 'V4', 'V5', "OFFST"]] = dfacc[['BOROUGH', 'LOC', 'ZIP','ONST', 'CRST', 'CFV1',
       'CFV2', 'CFV3',
       'CFV4', 'CFV5', 'V1', 'V2',
       'V3', 'V4', 'V5', "OFFST"]].astype("category")

#%%
#%%
dfacc.info()
# %%
dfacc.describe().round(2)
#%%
# ERG Note: Date and time ... creating new column combining date and time as pandas datetime type
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

#%%
dfacc["monsince"] = 12*(dfacc["YEAR"] - 2012) + (dfacc["MONTH"] - 7)
dfacc["daysince"] = 365*(dfacc["YEAR"] - 2012) + (dfacc["MONTH"] - 7)*30 + (dfacc["DAY"] - 1)


# %%
# add a column for month and year
dfacc['monyear'] = dfacc['DATETIME'].dt.to_period('M')

dfacc['monyear'] =dfacc['monyear'].astype("datetime64[ns]") 
# add a column with value one to permit counting of accidents later.
dfacc["COUNT"] = 1
#dfacc_monyear_harms = pd.DataFrame(dfacc[["monyear", "NUMINJ", "NUMKIL"]])

#%%
# Adding columns with boolean for whether there were injuries and whether their were deaths
# and where there were any harms
dfacc["INJBOL"] = dfacc["NUMINJ"] + dfacc["NUMPEDINJ"] + dfacc["NUMCYCINJ"] + dfacc["NUMMOTINJ"] > 0
dfacc["KILBOL"] = dfacc["NUMKIL"] + dfacc["NUMPEDKIL"] + dfacc["NUMCYCKIL"] + dfacc["NUMMOTKIL"] > 0
dfacc["HARMBOL"] = (dfacc["INJBOL"] > 0) | (dfacc["KILBOL"] > 0)
#%%
print(dfacc.info())

#%%
dfacc.head()

#%%
dfacc.describe()
#%%
# Distribution of number of deaths
plt.figure(figsize=(12, 6))
#sns.set(font_scale=1.8)
sns.histplot(data = dfacc, hue = "BOROUGH", x = "NUMKIL", multiple = "stack")
plt.show()

# Most accidents don't lead to deaths. No accident caused more than 8 deaths. (Assuming the records are complete + accurate.)


#%%
# average number of deaths per accident by BOROUGH
plt.figure(figsize=(12, 6))
deathsper_bor = sns.barplot(data = dfacc, hue = "BOROUGH", x = "BOROUGH", y = "NUMKIL", legend = False)
#deathsper_bor.bar_label(deathsper_bor.containers[0], fontsize=14)
plt.title('Average Number of People Killed', fontsize = 20)
plt.xlabel('Borough',  fontsize=16)
plt.ylabel('Average Number Killed', fontsize=16)
plt.show()
#%%
# #%%
# average number of deaths per accident by YEAR
sns.barplot(data = dfacc, hue = "BOROUGH", x = "YEAR", y = "NUMKIL", legend = False) 

#%%
# distribution of number of injuries
sns.histplot(data = dfacc, hue = "BOROUGH", x = "NUMINJ", multiple = "stack")
# 
#%%
# average number of people injured per accident
plt.figure(figsize=(12, 6))
sns.barplot(data = dfacc, hue = "BOROUGH", x = "BOROUGH", y = "NUMINJ", legend = False)
#deathsper_bor.bar_label(deathsper_bor.containers[0], fontsize=14)
plt.title('Average Number of People Injured', fontsize = 20)
plt.xlabel('Borough',  fontsize=16)
plt.ylabel('Average Number Injured', fontsize=16)
# 

#%%
sns.histplot(data = dfacc, x = "BOROUGH", hue = "BOROUGH", legend = False)
# Over the full time window, Brooklyn has had the most accidents, 
# followed by Queens, Manhattan, the Bronx, and Statten Island.

#%%
plt.figure(figsize=(12, 6))
sns.histplot(data = dfacc, bins = 136, hue = "BOROUGH", x = "DATETIME", multiple = "stack")
plt.title('Distribution of Accidents Over Time', fontsize = 20)
plt.xlabel('Date',  fontsize=16)
plt.ylabel('Number of Accidents', fontsize=16)
# There is a clear drop in # accidents after pandemic onset!
# There does appear to be a cyclical variation in the number of accidents!
# Hmm... but what are we actually seeing?

#%%

#df_grp = dfacc.groupby(['monyear', "BOROUGH"])['COUNT'].sum().reset_index()

#plt.figure(figsize=(12, 6))
#sns.barplot(data = df_grp, hue = "BOROUGH", x = "monyear", y = "COUNT", multiple = "stack")
#plt.title('Distribution of Accidents Over Time', fontsize = 20)
#plt.xlabel('Date',  fontsize=16)
#plt.ylabel('Number of Accidents', fontsize=16)
#%%
#%%
sns.histplot(data = dfacc, bins = 136, hue = "BOROUGH", x = "DATETIME", multiple = "stack")
plt.xlabel("Date")
plt.title("Distribution of Accidents Reported Over Time")

#%%
#count number of deaths per year for each borough
dfacc_yearly_bor = dfacc.groupby(["YEAR", "BOROUGH"]).NUMKIL.sum().reset_index()

#%%
sns.barplot(data = dfacc_yearly_bor, hue = "BOROUGH", x = "YEAR", y = "NUMKIL"
, legend = False)

#%%
#count number of injuries per year for each borough
dfacc_yearly = dfacc.groupby(["YEAR"]).COUNT.sum().reset_index()
dfacc_yearly_inj = dfacc.groupby(["YEAR"]).NUMINJ.sum().reset_index()
dfacc_yearly_kil = dfacc.groupby(["YEAR"]).NUMKIL.sum().reset_index()

dfacc_yearly["NUMINJ"] = dfacc_yearly_inj["NUMINJ"]
dfacc_yearly["NUMKIL"] = dfacc_yearly_kil["NUMKIL"]

#%%

plt.figure(figsize=(12, 6))
#sns.set(font_scale=1.8)
countyear = sns.barplot(data=dfacc_yearly, x="YEAR", y='COUNT', color = "green")
countyear.bar_label(countyear.containers[0], fontsize=14)
plt.title('Number of NYC Traffic Accidents Each Year', fontsize=24)
plt.xlabel('Year',  fontsize=16)
plt.ylabel('Number of Accidents', fontsize=16)
plt.show()
#%%

plt.figure(figsize=(12, 6))
#sns.set(font_scale=1.8)
injyear = sns.barplot(data=dfacc_yearly, x="YEAR", y='NUMINJ', color = "blue")
injyear.bar_label(injyear.containers[0], fontsize=14)
plt.title('Number of People Injured Annually in NYC Traffic Accidents', fontsize=24)
plt.xlabel('Year',  fontsize=16)
plt.ylabel('Number of People Injured', fontsize=16)
plt.show()
#%%

dfacc_yearly_bol = dfacc.groupby(["YEAR", "BOROUGH"]).HARMBOL.sum().reset_index()

plt.figure(figsize=(12, 6))
#sns.set(font_scale=1.8)
injyear = sns.barplot(data=dfacc_yearly_bol, x="YEAR", y='HARMBOL', color = "blue")
injyear.bar_label(injyear.containers[0], fontsize=14)
plt.title('Number of NYC Traffic Accidents Resulting in Death or Injury Annually', fontsize=24)
plt.xlabel('Year',  fontsize=16)
plt.ylabel('Number of Accidents', fontsize=16)
plt.show()

#%%

plt.figure(figsize=(12, 6))
#sns.set(font_scale=1.8)
kilyear = sns.barplot(data=dfacc_yearly, x="YEAR", y='NUMKIL', color = "red")
kilyear.bar_label(kilyear.containers[0], fontsize=14)
plt.title('Number of People Killed Annually in NYC Traffic Accidents', fontsize=24)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Number of People Killed', fontsize=16)
plt.show()


#%%
# Let's look by month: 
testt = sns.histplot(data = dfacc, x = "MONTH", hue = "BOROUGH", multiple = "stack")
sns.move_legend(
    testt, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
)

# Seems to be... we should check whether this is statistically significant. (That is someone else's research question!)
# But I wonder if the transition to COVID Lockdown has an influence here? 

#%%
# Let's look before COVID lockdown (roughly defined as before 2020)
dfacc_bef = dfacc.loc[(dfacc["YEAR"] < 2020)]
test2 = sns.histplot(data = dfacc_bef, hue = "BOROUGH", x = "MONTH", multiple = "stack")
sns.move_legend(test2, "upper left", bbox_to_anchor=(1, 1))
# Seems a bit more pronounced. Worth testing further... later!

#%%
# Let's look after COVID lockdown (after 2020)
dfacc_aft = dfacc.loc[(dfacc["YEAR"] > 2020)]
test3 = sns.histplot(data = dfacc_aft, hue = "BOROUGH", x = "MONTH", multiple = "stack")
sns.move_legend(test3, "upper left", bbox_to_anchor=(1, 1))
# but note we don't have full data for November or December.

#%%

# Now for some temporal trends!
# NUMINJ in auto accidents over time
sns.scatterplot(x = "DATETIME", y = "NUMINJ",
             hue = "BOROUGH", data = dfacc)

#%%
sns.scatterplot(x = "DATETIME", y = "NUMKIL",
             hue = "BOROUGH", data = dfacc)

#%%
# New efforts 11/25/2023

#dfacc["INJBOL"] = dfacc["NUMINJ"] > 0
#dfacc["KILBOL"] = dfacc["NUMKIL"] > 0

#%%
# Creating monthly version of original dfacc
dfacc_monthly = dfacc.groupby(["monyear", "BOROUGH"]).COUNT.sum().reset_index()

#%%
# creating monthly for each of the elements of interest by borough 
dfacc_monthly["NUMINJ"] = dfacc.groupby(["monyear", "BOROUGH"]).NUMINJ.sum().reset_index()["NUMINJ"]
dfacc_monthly["NUMACCINJ"] = dfacc.groupby(["monyear", "BOROUGH"]).INJBOL.sum().reset_index()["INJBOL"]
dfacc_monthly["NUMKIL"] = dfacc.groupby(["monyear", "BOROUGH"]).NUMKIL.sum().reset_index()["NUMKIL"]
dfacc_monthly["NUMACCKIL"] = dfacc.groupby(["monyear", "BOROUGH"]).KILBOL.sum().reset_index()["KILBOL"]
dfacc_monthly["NUMACCHRM"] = dfacc.groupby(["monyear", "BOROUGH"]).HARMBOL.sum().reset_index()["HARMBOL"]

#%%
# And now adding these columns to the monthly df
dfacc_monthly["SHAREACCINJ"] = dfacc_monthly["NUMACCINJ"] / dfacc_monthly["COUNT"]
dfacc_monthly["SHAREACCKIL"] = dfacc_monthly["NUMACCKIL"] / dfacc_monthly["COUNT"]  
dfacc_monthly["SHAREACCHRM"] = dfacc_monthly["NUMACCHRM"] / dfacc_monthly["COUNT"]  


#%%
# Converting back to datetime
dfacc_monthly["monyear"] = dfacc_monthly["monyear"].astype("datetime64[ns]")

#%%
# Plotting number of accidents each month that yield injury.
sns.scatterplot(data = dfacc_monthly, hue = "BOROUGH", x = "monyear", y = "NUMACCINJ", legend = False)
#%%
# Plotting number of accidents each month that yield deaths.
sns.scatterplot(data = dfacc_monthly, hue = "BOROUGH", x = "monyear", y = "NUMACCKIL", legend = False)

#%%
# # Plotting number of accidents each month that yield injury or death.
sns.scatterplot(data = dfacc_monthly, hue = "BOROUGH", x = "monyear", y = "NUMACCHRM", legend = False)


#%%
sns.histplot(data = dfacc_monthly, x="NUMACCINJ")

#%%
sns.histplot(data = dfacc_monthly, x="NUMACCKIL")
#%%
# Plotting share of accidents each month that yield injury.
sns.scatterplot(data = dfacc_monthly, hue = "BOROUGH", x = "monyear", y = "SHAREACCINJ")

#%%
# Plotting share of accidents each month that yield death.
sns.scatterplot(data = dfacc_monthly, hue = "BOROUGH", x = "monyear", y = "SHAREACCKIL")

#%%
# looking at NUMACCINJ over time

sns.scatterplot(x = "monyear", y = "NUMACCINJ",
             hue = "BOROUGH", data = dfacc_monthly)

#%%
# looking at NUMACCHRM over time

#sns.scatterplot(x = "monyear", y = "NUMACCHRM",
#             hue = "BOROUGH", data = dfacc_monthly)

#%%
dfacc_monthly["YEAR"] = dfacc_monthly["monyear"].dt.year
dfacc_monthly["MONTH"] = dfacc_monthly["monyear"].dt.month

dfacc_monthly["YEAR"] = dfacc_monthly["YEAR"].astype("category")
dfacc_monthly["MONTH"] = dfacc_monthly["MONTH"].astype("category")
#%%
dfacc_monthly["monyear"] = dfacc_monthly["monyear"].astype("datetime64[ns]")
#%%

acc = sns.histplot(data = dfacc, hue = "BOROUGH", x = "monyear", multiple = "stack")
#sns.move_legend(
#    acc, "upper center",
#    bbox_to_anchor=(1.2, 1), ncol=1, title=None, frameon=False,
#)
plt.title("Accidents Reported Over Time")
plt.show()

# something odd in the data here... is there somehow a data point missing?

# No -- it is not an issue with the data, it is an issue with the plot
# gaps increase when "hue" is removed

#%%

#%%
#Creating a new data frame that only includes accidents that resulted in injuries or deaths
dfacc_harm = dfacc.loc[(dfacc["NUMINJ"] > 0) | (dfacc["NUMKIL"] > 0)]

#%%
acc_harm = sns.histplot(data = dfacc_harm, hue = "BOROUGH", x = "monyear", multiple = "stack")
sns.move_legend(
    acc_harm, "upper center",
    bbox_to_anchor=(1.2, 1), ncol=1, title=None, frameon=False,
)
plt.title("Accidents Resulting in Injury or Death Reported Over Time")
plt.show()


#%%
#Creating a dataframe with monthly stats on the number of accidents resulting in harm

# Creating monthly version of original dfacc
#dfacc_monthly = dfacc.groupby(["monyear", "BOROUGH"]).COUNT.sum().reset_index()

# creating monthly for each of the elements of interest by borough 
#dfacc_monthly["NUMINJ"] = dfacc.groupby(["monyear", "BOROUGH"]).NUMINJ.sum().reset_index()["NUMINJ"]
#dfacc_monthly["NUMACCINJ"] = dfacc.groupby(["monyear", "BOROUGH"]).INJBOL.sum().reset_index()["INJBOL"]
#dfacc_monthly["NUMKIL"] = dfacc.groupby(["monyear", "BOROUGH"]).NUMKIL.sum().reset_index()["NUMKIL"]
#dfacc_monthly["NUMACCKIL"] = dfacc.groupby(["monyear", "BOROUGH"]).KILBOL.sum().reset_index()["KILBOL"]
#dfacc_monthly["NUMACCHRM"] = dfacc.groupby(["monyear", "BOROUGH"]).HARMBOL.sum().reset_index()["HARMBOL"]

#%%
#dfacc_monthly["SHAREACCINJ"] = dfacc_monthly["NUMACCINJ"] / dfacc_monthly["COUNT"]
#dfacc_monthly["SHAREACCKIL"] = dfacc_monthly["NUMACCKIL"] / dfacc_monthly["COUNT"]  
#dfacc_monthly["SHAREACCHRM"] = dfacc_monthly["NUMACCHRM"] / dfacc_monthly["COUNT"]  


#%%
# Converting back to datetime
#dfacc_monthly["monyear"] = dfacc_monthly["monyear"].astype("datetime64[ns]")

#%%
#dfacc_monthly["year"] = dfacc_monthly["monyear"].dt.year
#dfacc_monthly["month"] = dfacc_monthly["monyear"].dt.month


#%%
dfacc_monthly["monsince"] = 12*(dfacc_monthly["YEAR"].astype("int64") - 2012) + (dfacc_monthly["MONTH"].astype("int64") - 7)

#%%
# Comparing # of deaths pre- and post-
#ttest_kil = scipy.stats.ttest_ind(dfacc_harm_pre["NUMKIL"], dfacc_harm_post["NUMKIL"])
#print(ttest_kil)


#%%
import scipy
dfacc_monthly_BRONX = dfacc_monthly.loc[(dfacc_monthly["BOROUGH"] == "BRONX")]
dfacc_monthly_BROOKLYN = dfacc_monthly.loc[(dfacc_monthly["BOROUGH"] == "BROOKLYN")]
dfacc_monthly_MANHATTAN = dfacc_monthly.loc[(dfacc_monthly["BOROUGH"] == "MANHATTAN")]
dfacc_monthly_STATEN_ISLAND = dfacc_monthly.loc[(dfacc_monthly["BOROUGH"] == "STATEN ISLAND")]
dfacc_monthly_QUEENS = dfacc_monthly.loc[(dfacc_monthly["BOROUGH"] == "QUEENS")]

#%%
#Finding the value for monsince corresponding to March 2020
print(dfacc_monthly.loc[(dfacc_monthly["YEAR"] == 2020) & (dfacc_monthly["MONTH" ] == 3) ]["monsince"])

#%%

# Assuming that March and April will be anomalous

# window before March, 2023
dfacc_monthly_pre = dfacc_monthly.loc[(dfacc_monthly["monsince"] < 92)]
# window after April, 2023
dfacc_monthly_post = dfacc_monthly.loc[(dfacc_monthly["monsince"] > 93)]

# window before March, 2023
dfacc_pre = dfacc.loc[(dfacc["monsince"] < 92)]
# window after April, 2023
dfacc_post = dfacc.loc[(dfacc["monsince"] > 93)]

#%%
dfacc_monthly_pre["MONTH"]= dfacc_monthly_pre["MONTH"].astype("category")
dfacc_monthly_post["MONTH"]= dfacc_monthly_post["MONTH"].astype("category")
#%%

dfacc_monthly_pre_total = dfacc_monthly_pre.groupby(["monsince", "BOROUGH"]).COUNT.sum().reset_index()
dfacc_monthly_pre_INJ = dfacc_monthly_pre.groupby(["monsince", "BOROUGH"]).NUMINJ.sum().reset_index()
dfacc_monthly_pre_KIL = dfacc_monthly_pre.groupby(["monsince", "BOROUGH"]).NUMKIL.sum().reset_index()
dfacc_monthly_pre_HRM = dfacc_monthly_pre.groupby(["monsince", "BOROUGH"]).NUMACCHRM.sum().reset_index()

dfacc_monthly_post_total = dfacc_monthly_post.groupby(["monsince", "BOROUGH"]).COUNT.sum().reset_index()
dfacc_monthly_post_INJ = dfacc_monthly_post.groupby(["monsince", "BOROUGH"]).NUMINJ.sum().reset_index()
dfacc_monthly_post_KIL = dfacc_monthly_post.groupby(["monsince", "BOROUGH"]).NUMKIL.sum().reset_index()
dfacc_monthly_post_HRM = dfacc_monthly_post.groupby(["monsince", "BOROUGH"]).NUMACCHRM.sum().reset_index()


#%%

#%%
# Sidebar: looking at one year before and one year after the pandemic

#dfacc_pre_t = dfacc_pre.groupby(["monsince"]).sum()
#dfacc_post_t = dfacc_post.groupby(["monsince"]).sum()

#dfacc_pre_total = dfacc_pre.groupby(["monsince"]).COUNT.sum().reset_index()
#dfacc_pre_INJ = dfacc_pre.groupby(["monsince"]).NUMINJ.sum().reset_index()
#dfacc_pre_KIL = dfacc_pre.groupby(["monsince"]).NUMKIL.sum().reset_index()
#dfacc_pre_HRM = dfacc_pre.groupby(["monsince"]).HARMBOL.sum().reset_index()

#dfacc_pre_tot["COUNT"] = dfacc_pre_total["COUNT"]
#dfacc_pre_tot["COUNT"] = dfacc_pre_total["COUNT"]

#dfacc_post_total = dfacc_post.groupby(["monsince"]).COUNT.sum().reset_index()
#dfacc_post_INJ = dfacc_post.groupby(["monsince"]).NUMINJ.sum().reset_index()
#dfacc_post_KIL = dfacc_post.groupby(["monsince"]).NUMKIL.sum().reset_index()
#dfacc_post_HRM = dfacc_post.groupby(["monsince"]).HARMBOL.sum().reset_index()

#%%

#year_before["NUMINJ"] = dfacc_pre_INJ
#year_before["NUMKIL"] = dfacc_pre_KIL["NUMINJ"]
#year_before["NUMACCHRM"] = dfacc_pre_HRM["HARMBOL"]

#year_before["NUMINJ"] = dfacc_post_INJ
#year_before["NUMKIL"] = dfacc_post_KIL
#year_before["NUMACCHRM"] = dfacc_post_HRM

years_before = dfacc_pre[(dfacc_pre["monsince"] > 68)].groupby(["daysince"]).COUNT.sum().reset_index()
years_before["NUMACCHRM"] = dfacc_pre[(dfacc_pre["monsince"] > 68)].groupby(["daysince"]).HARMBOL.sum().reset_index()["HARMBOL"]
years_before["NUMINJ"] = dfacc_pre[(dfacc_pre["monsince"] > 68)].groupby(["daysince"]).NUMINJ.sum().reset_index()["NUMINJ"]
years_before["NUMKIL"] = dfacc_pre[(dfacc_pre["monsince"] > 68)].groupby(["daysince"]).NUMKIL.sum().reset_index()["NUMKIL"]

years_after = dfacc_post[(dfacc_post["monsince"] < 117)].groupby(["daysince"]).COUNT.sum().reset_index()
years_after["NUMACCHRM"] = dfacc_post[(dfacc_post["monsince"] < 117)].groupby(["daysince"]).HARMBOL.sum().reset_index()["HARMBOL"]
years_after["NUMINJ"] = dfacc_post[(dfacc_post["monsince"] < 117)].groupby(["daysince"]).NUMINJ.sum().reset_index()["NUMINJ"]
years_after["NUMKIL"] = dfacc_post[(dfacc_post["monsince"] < 117)].groupby(["daysince"]).NUMKIL.sum().reset_index()["NUMKIL"]
#%%

#%%
#sns.histplot(data = year_before, x = "monsince")
#%%
sns.histplot(data = years_before, x = "COUNT")

#Approximately normally distributed
#%%
sns.histplot(data = years_after, x = "COUNT")


#%%
#SidebarTTEST
ttest_numacc = scipy.stats.ttest_ind(years_before["COUNT"], years_after["COUNT"])
print(ttest_numacc)
# approximately normally distributed

#%%
years_before.describe()

#%%
years_after.describe()

#%%
sns.histplot(data = years_before, x = "NUMACCHRM")

#Approximately normally distributed
#%%
sns.histplot(data = years_after, x = "NUMACCHRM")

#%%
ttest_numacchrm = scipy.stats.ttest_ind(years_before["NUMACCHRM"], years_after["NUMACCHRM"])
print(ttest_numacchrm)

#%%
ttest_numkil = scipy.stats.ttest_ind(years_before["NUMKIL"], years_after["NUMKIL"])
print(ttest_numkil)

#%%
ttest_numinj = scipy.stats.ttest_ind(years_before["NUMINJ"], years_after["NUMINJ"])
print(ttest_numinj)

#%%

# aggregating across borough
#dfacc_monthly_pre_total = dfacc_monthly_pre.groupby(["monsince"]).COUNT.sum().reset_index()
#dfacc_monthly_pre_INJ = dfacc_monthly_pre.groupby(["monsince"]).NUMINJ.sum().reset_index()
#dfacc_monthly_pre_KIL = dfacc_monthly_pre.groupby(["monsince"]).NUMKIL.sum().reset_index()

#dfacc_monthly_post_total = dfacc_monthly_post.groupby(["monsince"]).COUNT.sum().reset_index()
#dfacc_monthly_post_INJ = dfacc_monthly_post.groupby(["monsince"]).NUMINJ.sum().reset_index()
#dfacc_monthly_post_KIL = dfacc_monthly_post.groupby(["monsince"]).NUMKIL.sum().reset_index()

#%%
# #
#%%
#########################
# Modeling number of accidents pre- March 2020
import statsmodels.formula.api as smf
lm = smf.ols(formula='NUMACCHRM ~ daysince', data = years_before).fit()
print(lm.summary())

#%%
# Modeling number of injuries post- April 2020
import statsmodels.formula.api as smf
lm = smf.ols(formula='NUMACCHRM ~ daysince', data=years_after).fit()
print(lm.summary())

#%%
frames = [years_before, years_after]

combined = pd.concat(frames)

#%%
sns.scatterplot(data = combined, x = "daysince", y = "NUMACCHRM")
plt.title('Daily Number of Accidents resulting in Death or Injury', fontsize = 20)
plt.xlabel('Day',  fontsize=16)
plt.ylabel('Number of Accidents', fontsize=16)
plt.show()

#%%
sns.regplot(data = years_before, x = "daysince", y = "NUMACCHRM", color = "green", marker = ".")
#plt.title('Daily Number of Accidents resulting in Death or Injury', fontsize = 20)
plt.xlabel('Day',  fontsize=16)
plt.ylabel('Number of Accidents', fontsize=16)
plt.ylim(0, 350)
plt.show()

#%%
sns.regplot(data = years_after, x = "daysince", y = "NUMACCHRM", color = "red", marker = ".")
#plt.title('Daily Number of Accidents resulting in Death or Injury', fontsize = 20)
plt.xlabel('Day',  fontsize=16)
plt.ylabel('Number of Accidents', fontsize=16)
plt.ylim(0, 350)
plt.show()

#%%
# Modeling number of injuries pre- March 2020
import statsmodels.formula.api as smf
lm = smf.ols(formula='NUMINJ ~ monsince', data = dfacc_monthly_pre_INJ).fit()
print(lm.summary())

#%%
# Modeling number of injuries post- April 2020
import statsmodels.formula.api as smf
lm = smf.ols(formula='NUMINJ ~ monsince', data=dfacc_monthly_post_INJ).fit()
print(lm.summary())

#%%
#########################
# Modeling number of accidents pre- March 2020
import statsmodels.formula.api as smf
lm = smf.ols(formula='COUNT ~ monsince', data = dfacc_monthly_pre_total).fit()
print(lm.summary())

#%%
# Modeling number of injuries post- April 2020
import statsmodels.formula.api as smf
lm = smf.ols(formula='COUNT ~ monsince', data=dfacc_monthly_post_total).fit()
print(lm.summary())

#%%
# Modeling number of injuries pre- March 2020
import statsmodels.formula.api as smf
lm = smf.ols(formula='NUMINJ ~ monsince', data = dfacc_monthly_pre_INJ).fit()
print(lm.summary())

#%%
# Modeling number of injuries post- April 2020
import statsmodels.formula.api as smf
lm = smf.ols(formula='NUMINJ ~ monsince', data=dfacc_monthly_post_INJ).fit()
print(lm.summary())

#%%
sns.scatterplot(data = dfacc_monthly_pre, x = "monsince", y = dfacc_monthly_pre.groupby(["monsince"]).COUNT.sum().reset_index()["COUNT"], legend = False)

#%%

## Problem here... why??
sns.scatterplot(data = dfacc_monthly_post, x = "monsince", y = dfacc_monthly_post.groupby(["monsince"]).COUNT.sum().reset_index()["COUNT"], legend = False)
plt.show()

#%%
import statsmodels.formula.api as smf
lm = smf.ols(formula='NUMINJ ~ monsince', data=dfacc_monthly).fit()
print(lm.summary())

#%%
import statsmodels.formula.api as smf
lm = smf.ols(formula='NUMINJ ~ monsince', data=dfacc_monthly_pre).fit()
print(lm.summary())

#%%
#TTEST
ttest_1 = scipy.stats.ttest_ind(dfacc_monthly_BRONX["COUNT"], dfacc_monthly_BROOKLYN["COUNT"])
print(ttest_1)

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
#dfacc_pre_g_tot = dfacc_pre.groupby(["monyear", "BOROUGH"]).COUNT.count().reset_index()
#dfacc_post_g_tot = dfacc_post.groupby(["monyear", "BOROUGH"]).COUNT.count().reset_index()

#%%
#dfacc_pre_g["TOT"] = dfacc_pre_g_tot["COUNT"]
#dfacc_pre_g["SHAREINJ"] = dfacc_pre_g["NUMINJ"] / dfacc_pre_g["TOT"] 
#dfacc_post_g["TOT"] = dfacc_post_g_tot["COUNT"]
#dfacc_post_g["SHAREINJ"] = dfacc_post_g["NUMINJ"] / dfacc_post_g["TOT"]


#%%

# Converting back to datetime
#dfacc_pre_g["monyear"] = dfacc_pre_g["monyear"].astype("datetime64[ns]")
#dfacc_post_g["monyear"] = dfacc_post_g["monyear"].astype("datetime64[ns]")

#%%
#sns.histplot(data = dfacc_post, hue = "BOROUGH", x = "monyear")

#%%
# Plotting share of accidents each month that yield injury.
#sns.scatterplot(x = "monyear", y = "SHAREINJ",
             hue = "BOROUGH", data = dfacc_pre_g)

#%%
# Plotting share of accidents each month that yield injury.
sns.scatterplot(x = "monyear", y = "SHAREINJ",
             hue = "BOROUGH", data = dfacc_post_g)

#%%
#Linear models

import statsmodels.formula.api as smf
lm = smf.ols(formula='NUMINJ ~ BOROUGH', data=dfacc_monthly).fit()
print(lm.summary())

#%%
#for i in dfacc_monthly["monyear"]:
#    dfacc_monthly["monyear"][i] = dfacc_monthly["monyear"][i].delta.month()
#dfacc_monthly["monyear"] = pd.to_datetime(dfacc_monthly["monyear"]) - pd.datetime(2012,7,1)

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
