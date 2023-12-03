#%% [markdown]

# Team 6 Project Code

# Emily's data cleaning and variable renaming for Team use

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#import sklearn
from sklearn import linear_model
# %% [markdown]

# --> ERG Note to team: rename csv as "acc.csv"

#dfacc = pd.read_csv('acc.csv')
# %%

dfacc = pd.read_csv('C:/Users/saira/OneDrive/Desktop/GWU Courses/Into to Data Mining/Project/Datasets/Crashes.csv')

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
dfacc["HOUR"] = dfacc["DATETIME"].dt.hour

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
plt.xticks(rotation=45)
plt.title('Histogram of contributing factor for vehicle 1')
plt.show()


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
plt.title("Total number of people injured by season")
plt.show()

#%%[markdown]

# Are accidents more deadlier and dangerous in different seasons ?

# %%

by_season = full_year.groupby(['SEASON', 'YEAR'])[['NUMINJ', 'NUMKIL']].mean().reset_index()


plt.figure(figsize=(10, 6))
sns.lineplot(data=by_season,x='YEAR',y='NUMINJ',hue='SEASON')
plt.title("Average number of people injured by season")
plt.show()


plt.figure(figsize=(10, 6))
sns.lineplot(data=by_season,x='YEAR',y='NUMKIL',hue='SEASON')
plt.title("Average number of people killed by season")
plt.show()


#%%[markdown]

# Are accidents more deadlier and dangerous at different times ?




# %%

by_hour = full_year.groupby(['HOUR'])[['NUMINJ', 'NUMKIL','NUMPEDINJ', 'NUMPEDKIL', 'NUMCYCINJ',
       'NUMCYCKIL', 'NUMMOTINJ', 'NUMMOTKIL']].mean().reset_index()


plt.figure(figsize=(10, 6))
sns.barplot(data=by_hour,x='HOUR',y='NUMINJ')
plt.title("Average number of people injured by hour")
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(data=by_hour,x='HOUR',y='NUMKIL')
plt.title("Average number of people killed by hour")
plt.show()


#%%
plt.figure(figsize=(10, 6))
sns.lineplot(data=by_hour, x='HOUR', y='NUMPEDINJ', label='Pedestrian Injuries')
sns.lineplot(data=by_hour, x='HOUR', y='NUMCYCINJ', label='Cyclist Injuries')
sns.lineplot(data=by_hour, x='HOUR', y='NUMMOTINJ', label='Motorist Injuries')
plt.legend()
plt.title("Average Number of People Injured by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Number of Injuries")
plt.show()
# %%
plt.figure(figsize=(15, 10))

# Subplot 1: NUMPEDINJ
plt.subplot(2, 2, 1)
sns.barplot(data=by_hour, x='HOUR', y='NUMPEDINJ', color='skyblue')
plt.title('Pedestrian Injuries')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Number of Injuries')

# Subplot 2: NUMCYCINJ
plt.subplot(2, 2, 2)
sns.barplot(data=by_hour, x='HOUR', y='NUMCYCINJ', color='lightcoral')
plt.title('Cyclist Injuries')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Number of Injuries')

# Subplot 3: NUMMOTINJ
plt.subplot(2, 2, 3)
sns.barplot(data=by_hour, x='HOUR', y='NUMMOTINJ', color='lightgreen')
plt.title('Motorist Injuries')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Number of Injuries')

# Subplot 4: NUMINJ
plt.subplot(2, 2, 4)
sns.barplot(data=by_hour, x='HOUR', y='NUMINJ', color='orange')
plt.title('Total Injuries')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Number of Injuries')

plt.tight_layout()
plt.show()

#%%[markdown]
#Same thing but totals instead of averages
# %%

by_hour = full_year.groupby(['HOUR'])[['NUMINJ', 'NUMKIL','NUMPEDINJ', 'NUMPEDKIL', 'NUMCYCINJ',
       'NUMCYCKIL', 'NUMMOTINJ', 'NUMMOTKIL']].sum().reset_index()

plt.figure(figsize=(15, 10))

# Subplot 1: NUMPEDINJ
plt.subplot(2, 2, 1)
sns.barplot(data=by_hour, x='HOUR', y='NUMPEDINJ', color='skyblue')
plt.title('Pedestrian Injuries')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Number of Injuries')

# Subplot 2: NUMCYCINJ
plt.subplot(2, 2, 2)
sns.barplot(data=by_hour, x='HOUR', y='NUMCYCINJ', color='lightcoral')
plt.title('Cyclist Injuries')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Number of Injuries')

# Subplot 3: NUMMOTINJ
plt.subplot(2, 2, 3)
sns.barplot(data=by_hour, x='HOUR', y='NUMMOTINJ', color='lightgreen')
plt.title('Motorist Injuries')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Number of Injuries')

# Subplot 4: NUMINJ
plt.subplot(2, 2, 4)
sns.barplot(data=by_hour, x='HOUR', y='NUMINJ', color='orange')
plt.title('Total Injuries')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Number of Injuries')

plt.tight_layout()
plt.show()
# %%[markdown]

# Decision Tree Modelling

# Updating Severity clause and adding a variable to capture number of vehicles
conditions = [
    (df['NUMINJ'] == 0),
    (df['NUMINJ'] > 0) | (df['NUMKIL'] > 0)
]

values = ['No Injury', 'Injury']

df['SEVERITY'] = np.select(conditions, values, default='Others')


df['VEHICLES'] = df[['V1', 'V2', 'V3', 'V4', 'V5']].count(axis=1)


#%%

#%%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

features = ['BOROUGH', 'YEAR', 'MONTH', 'DAY', 'HOUR','VEHICLES']
target = 'SEVERITY' 

# Dropping missing data
df_model = df[features + [target]].dropna()

# Encoding Categories
#label_encoder = LabelEncoder()
df_model['BOROUGH'] = label_encoder.fit_transform(df_model['BOROUGH'])
#df_model['CFV1'] = label_encoder.fit_transform(df_model['CFV1'])

#  Creating Train Test split
X_train, X_test, y_train, y_test = train_test_split(
    df_model[features],
    df_model[target],
    test_size=0.2,
    random_state=42
)

clf = DecisionTreeClassifier(random_state=40, max_depth=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)




#%%



plt.figure(figsize=(20, 15))  # Adjust the figsize as needed

# Plot the decision tree with additional details
plot_tree(
    clf,
    filled=True,
    feature_names=features,
    rounded=True,
    class_names=clf.classes_,
    proportion=True,  # Show proportions in leaf nodes
    precision=2,  # Set precision for displayed values
    impurity=False,  # Remove impurity information for better clarity
    fontsize=10,  # Adjust fontsize for better readability
    node_ids=True,  # Show the IDs of the nodes
)

plt.show()








# %%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

target = 'SEVERITY'

# Dropping missing data
df_model = df[['BOROUGH', 'CFV1'] + [target]].dropna()

# One-hot encoding for categorical variables
df_model = pd.get_dummies(df_model, columns=['BOROUGH', 'CFV1'])

# List of all columns except the target variable
features = df_model.columns.difference([target]).tolist()

# Creating the Train Test split
X_train, X_test, y_train, y_test = train_test_split(
    df_model[features],
    df_model[target],
    test_size=0.2,
    random_state=42
)

# Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=40, max_depth=3)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)

# %%

%matplotlib inline
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Assuming clf is your decision tree classifier
plt.figure(figsize=(20, 10))  # Adjust the figsize as needed
plot_tree(clf, filled=True, feature_names=features, rounded=True)

plt.show()
# %%[markdown]
