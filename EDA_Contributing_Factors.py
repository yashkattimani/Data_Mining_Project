#%% [markdown]

# Team 6 Project Code

# Yash's data cleaning and variable renaming for Team use

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#import sklearn
from sklearn import linear_model
# %% [markdown]

# --> ERG Note to team: rename csv as "acc.csv"

dfacc = pd.read_csv("C:\\Users\\YASH KATTIMANI\\Downloads\\acc.csv")
# %%

#dfacc = pd.read_csv('C:/Users/saira/OneDrive/Desktop/GWU Courses/Into to Data Mining/Project/Datasets/Crashes.csv')

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
# Changing columns names

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

# ERG Note: Here is the mapping of original variables to the new

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

#%% [markdown]

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

#%%

# ERG Note: Adding separate columns for year, month, and day

dfacc["YEAR"] = dfacc["DATETIME"].dt.year
dfacc["MONTH"] = dfacc["DATETIME"].dt.month
dfacc["DAY"] = dfacc["DATETIME"].dt.day

#%%
print(dfacc.info())

#%% [markdown]

# Exploratory Data Analysis


#%%


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
df=dfacc.copy()

df.head(15)


# %%
plt.figure(figsize=(20, 8))

factor_counts = df['CFV1'].value_counts().sort_values(ascending=False)

# Histogram
sns.countplot(data=df, x='CFV1', order=factor_counts.index)
plt.xticks(rotation=85)
plt.title('Histogram of contributing factor for vehicle 1')
plt.show()

#%%
# Perform value counts on 'CFV1' column
cfv1_value_counts = df['CFV1'].value_counts()

# Display the result
print(cfv1_value_counts.to_string())

#%%
# Perform value counts on 'CFV1' column
cfv1_value_counts = df['CFV1'].value_counts()

# Create a new column with proportions
df['CFV1_Proportion'] = df['CFV1'].map(lambda x: round((cfv1_value_counts.get(x, 0) / len(df)) * 100, 2) if pd.notna(x) else 0)

# Display the result in descending order
result = df[['CFV1', 'CFV1_Proportion']].drop_duplicates().sort_values('CFV1_Proportion', ascending=False)

# Print the rounded proportions
print(result.to_string(index=False))


#%%
# Perform value counts on 'CFV1' column excluding 'Unspecified'
cfv1_value_counts = df[df['CFV1'] != 'Unspecified']['CFV1'].value_counts()

# Create a new column with proportions
df['CFV1_Proportion'] = df['CFV1'].map(lambda x: round((cfv1_value_counts.get(x, 0) / len(df[df['CFV1'] != 'Unspecified'])) * 100, 2) if pd.notna(x) else 0)

# Display the result in descending order
result = df[df['CFV1'] != 'Unspecified'][['CFV1', 'CFV1_Proportion']].drop_duplicates().sort_values('CFV1_Proportion', ascending=False)

# Print the rounded proportions
print(result.to_string(index=False))

#%% [markdown]
# Major Contributing factors for all Vehicles 
# 1. Driver Inattention/Distraction: 30.30%
# 2. Failure to Yield Right-of-Way: 8.99%
# 3. Following Too Closely: 8.11%
# 4. Backing Unsafely: 5.64%
# 5. Other Vehicular: 4.72%
# 6. Passing or Lane Usage Improper: 4.20%
# 7. Passing Too Closely: 3.77%
# 8. Turning Improperly: 3.76%
# 9. Fatigued/Drowsy: 3.53%
# 10. Unsafe Lane Changing: 2.99%

#%%
# plot for Contributing factors for Vehicle 2
plt.figure(figsize=(20, 8))

factor_counts = df['CFV2'].value_counts().sort_values(ascending=False)

# Histogram
sns.countplot(data=df, x='CFV2', order=factor_counts.index)
plt.xticks(rotation=85)
plt.title('Histogram of contributing factor for vehicle 2')
plt.show()

#%%
# Perform value counts on 'CFV2' column excluding 'Unspecified'
cfv2_value_counts = df[df['CFV2'] != 'Unspecified']['CFV2'].value_counts()

# Create a new column with proportions for 'CFV2'
df['CFV2_Proportion'] = df['CFV2'].map(lambda x: round((cfv2_value_counts[x] / len(df[df['CFV2'] != 'Unspecified'])) * 100, 2))

# Display the result for 'CFV2' in descending order
result_cfv2 = df[df['CFV2'] != 'Unspecified'][['CFV2', 'CFV2_Proportion']].drop_duplicates().sort_values('CFV2_Proportion', ascending=False)

# Print the rounded proportions for 'CFV2'
print(result_cfv2.to_string(index=False))

#%%
# Plot for Contributing factors for Vehicle 3
plt.figure(figsize=(20, 8))

factor_counts_cfv3 = df['CFV3'].value_counts().sort_values(ascending=False)

# Histogram
sns.countplot(data=df, x='CFV3', order=factor_counts_cfv3.index)
plt.xticks(rotation=85)
plt.title('Histogram of contributing factor for vehicle 3')
plt.show()
#%%
cfv3_value_counts = df[df['CFV3'] != 'Unspecified']['CFV3'].value_counts()

# Create a new column with proportions for 'CFV3'
df['CFV3_Proportion'] = df['CFV3'].map(lambda x: round((cfv3_value_counts.get(x, 0) / len(df[df['CFV3'] != 'Unspecified'])) * 100, 2) if pd.notna(x) else 0)

# Display the result for 'CFV3' in descending order
result_cfv3 = df[df['CFV3'] != 'Unspecified'][['CFV3', 'CFV3_Proportion']].drop_duplicates().sort_values('CFV3_Proportion', ascending=False)

# Print the rounded proportions for 'CFV3'
print(result_cfv3.to_string(index=False))
#%%
# Plot for Contributing factors for Vehicle 4
plt.figure(figsize=(20, 8))

factor_counts_cfv4 = df['CFV4'].value_counts().sort_values(ascending=False)

# Histogram
sns.countplot(data=df, x='CFV4', order=factor_counts_cfv4.index)
plt.xticks(rotation=85)
plt.title('Histogram of contributing factor for vehicle 4')
plt.show()

#%%
# Perform value counts on 'CFV4' column excluding 'Unspecified'
cfv4_value_counts = df[df['CFV4'] != 'Unspecified']['CFV4'].value_counts()

# Create a new column with proportions for 'CFV4'
df['CFV4_Proportion'] = df['CFV4'].map(lambda x: round((cfv4_value_counts.get(x, 0) / len(df[df['CFV4'] != 'Unspecified'])) * 100, 2))

# Display the result for 'CFV4' in descending order
result_cfv4 = df[df['CFV4'] != 'Unspecified'][['CFV4', 'CFV4_Proportion']].drop_duplicates().sort_values('CFV4_Proportion', ascending=False)

# Print the rounded proportions for 'CFV4'
print(result_cfv4.to_string(index=False))


#%%
# Filter relevant columns
injury_death_data = df[['CFV1', 'CFV2', 'NUMINJ', 'NUMKIL', 'BOROUGH']]

# Combine both contributing factors into a single column
injury_death_data['CONTRIBUTING_FACTOR'] = injury_death_data['CFV1'].fillna(injury_death_data['CFV2'])

# Drop original contributing factor columns
injury_death_data = injury_death_data[['CONTRIBUTING_FACTOR', 'NUMINJ', 'NUMKIL', 'BOROUGH']]

# Group by contributing factor and borough, summing injuries and deaths
grouped_data = injury_death_data.groupby(['CONTRIBUTING_FACTOR', 'BOROUGH']).agg({'NUMINJ': 'sum', 'NUMKIL': 'sum'}).reset_index()

# Calculate proportions
grouped_data['INJURY_PROPORTION'] = grouped_data['NUMINJ'] / (grouped_data['NUMINJ'] + grouped_data['NUMKIL'])
grouped_data['DEATH_PROPORTION'] = grouped_data['NUMKIL'] / (grouped_data['NUMINJ'] + grouped_data['NUMKIL'])

# Sort by death proportion in descending order
sorted_data = grouped_data.sort_values('DEATH_PROPORTION', ascending=False)

# Display the result
print(sorted_data.to_string(index=False))

#%%
import pandas as pd

# Filter relevant columns
injury_death_data = df[['CFV1', 'CFV2', 'NUMINJ', 'NUMKIL', 'BOROUGH']]

# Combine both contributing factors into a single column
injury_death_data['CONTRIBUTING_FACTOR'] = injury_death_data['CFV1'].fillna(injury_death_data['CFV2'])

# Drop original contributing factor columns
injury_death_data = injury_death_data[['CONTRIBUTING_FACTOR', 'NUMINJ', 'NUMKIL', 'BOROUGH']]

# Group by contributing factor, borough, and sum injuries and deaths
grouped_data = injury_death_data.groupby(['CONTRIBUTING_FACTOR', 'BOROUGH']).agg({'NUMINJ': 'sum', 'NUMKIL': 'sum'}).reset_index()

# Group by borough and sum injuries and deaths
borough_grouped_data = injury_death_data.groupby('BOROUGH').agg({'NUMINJ': 'sum', 'NUMKIL': 'sum'}).reset_index()

# Merge with the grouped_data to calculate proportions
grouped_data = pd.merge(grouped_data, borough_grouped_data, on='BOROUGH', suffixes=('_contrib', '_borough'))

# Calculate proportions
grouped_data['INJURY_PROPORTION'] = grouped_data['NUMINJ_contrib'] / grouped_data['NUMINJ_borough']
grouped_data['DEATH_PROPORTION'] = grouped_data['NUMKIL_contrib'] / grouped_data['NUMKIL_borough']

# Sort by death proportion in descending order
sorted_data = grouped_data.sort_values('DEATH_PROPORTION', ascending=False)

# Display the result
print(sorted_data[['CONTRIBUTING_FACTOR', 'BOROUGH', 'INJURY_PROPORTION', 'DEATH_PROPORTION']].to_string(index=False))

# %%
factor_counts
# %%
df.columns
#%% [markdown]

### Classifying Severity 
#%%

conditions = [
    (df['NUMINJ'] == 0),
    (df['NUMINJ'] > 0) & (df['NUMKIL'] == 0),
    (df['NUMKIL'] > 0)
]

values = ['No Injury', 'Injury', 'Fatal']

df['SEVERITY'] = np.select(conditions, values, default='Others')

print(df[['NUMINJ', 'NUMKIL', 'SEVERITY']])

# %%
sns.countplot(data=df, x='SEVERITY', hue='YEAR',
              order=df['SEVERITY'].value_counts().index)

plt.title('Number of Each Category of Accident Severity by Year')
plt.xlabel('Accident Severity Category')
plt.ylabel('Count')
plt.legend(title='Year')

plt.show()
# %%
severity_counts = df.groupby(['YEAR', 'SEVERITY']).size().reset_index(name='Count')

years = severity_counts['YEAR'].unique()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

for i, year in enumerate(years):
    ax = axes[i // 3, i % 3]
    data = severity_counts[severity_counts['YEAR'] == year]
    labels = data['SEVERITY']
    sizes = data['Count']
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Year {year}')

plt.tight_layout()
plt.show()
# %%
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
severity_counts = df.groupby(['YEAR', 'SEVERITY']).size().reset_index(name='Count')

injury_data = severity_counts[severity_counts['SEVERITY'].isin(['Injury', 'No Injury'])]

fig, ax = plt.subplots(figsize=(10, 6))
sns.set_palette("pastel")

def update(frame):
    ax.clear()
    data = injury_data[injury_data['YEAR'] <= frame]
    sns.barplot(data=data, x='YEAR', y='Count', hue='SEVERITY', ax=ax)
    ax.set_title(f'Accident Severity Distribution ({frame})')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.legend(title='SEVERITY')

years = sorted(injury_data['YEAR'].unique())
ani = FuncAnimation(fig, update, frames=years, repeat=False)
animation_html = HTML(ani.to_jshtml())

animation_html

# %%
fig, ax = plt.subplots(figsize=(8, 6))
custom_palette = ['#A5D8DD', '#EA6A47']

def update(frame):
    ax.clear()
    data = injury_data[injury_data['Year'] <= frame]
    counts = data.groupby('Accident Severity Category')['Count'].sum()
    
    labels = counts.index
    sizes = counts.values
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'), colors=custom_palette)
    ax.set_title(f'Accident Severity Proportion ({frame})', pad=20)

years = sorted(injury_data['Year'].unique())
ani = FuncAnimation(fig, update, frames=years, repeat=False)

animation_html = HTML(ani.to_jshtml())

animation_html
# %%
by_factor_severity = df.groupby(['CFV1','SEVERITY']).count()
by_factor_severity.head(35)
by_factor_severity.reset_index(inplace=True)
# %%
total_counts = by_factor_severity.groupby('CFV1')['TIME'].sum()

#Total occurences
fatal_proportions = (
    by_factor_severity[by_factor_severity['SEVERITY'] == 'Injury']
    .groupby('CFV1')['TIME']
    .sum()
    / total_counts
).fillna(0)  

sorted_factors = fatal_proportions.sort_values(ascending=False)

top_10_factors = sorted_factors.head(10)
print(top_10_factors)

#%% [markdown]
#Seasonal/temporal analysis? More accidents in winter? During rush hour? Are they more deadly or more likely to result in injury at a different time? (Kush, Sairam)



# %%
df.head()
# %%
df['DATETIME'].max()
# %%
df['DATETIME'].min()

# %%
# removing 2012 and 2023 as they are incomplete years
full_year = df[(df['YEAR'] > 2012) & (df['YEAR'] < 2023)]
full_year['DATETIME'].min()
# %%
def get_season(month):
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Fall'
    else:
        return 'Winter'
#%%
full_year['SEASON'] = full_year['MONTH'].apply(get_season)
full_year[['MONTH','SEASON']].head(20)


# %%
by_season=full_year.groupby(['SEASON','YEAR'])[['NUMINJ','NUMKIL']].count()
# %%
by_season=full_year.groupby(['SEASON'])[['NUMINJ','NUMKIL']].count()
plt.figure(figsize=(10,8))
sns.barplot(by_season,x='SEASON',y='NUMINJ',hue='SEASON')
plt.title("Accident Occurence by Season")
plt.show()
# %%

by_season = full_year.groupby(['SEASON', 'YEAR'])[['NUMINJ', 'NUMKIL']].count().reset_index()


plt.figure(figsize=(10, 6))
sns.lineplot(data=by_season,x='YEAR',y='NUMINJ',hue='SEASON')
plt.show()

#%%
