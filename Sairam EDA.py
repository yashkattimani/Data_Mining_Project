#%%


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
file_path = "C:/Users/saira/OneDrive/Desktop/GWU Courses/Into to Data Mining/Project/Datasets/Crashes.csv"
df = pd.read_csv(file_path)

df.head(15)

# %%
df.shape
# %%
df.tail()
# %%
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
df['Year'] = df['CRASH DATE'].dt.year
df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'])
df['Hour'] = df['CRASH TIME'].dt.hour
# %%
plt.figure(figsize=(20, 8))

# Calculating the value counts in order to sort
factor_counts = df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().sort_values(ascending=False)

# Histogram
sns.countplot(data=df, x='CONTRIBUTING FACTOR VEHICLE 1', order=factor_counts.index)
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
    (df['NUMBER OF PERSONS INJURED'] == 0),
    (df['NUMBER OF PERSONS INJURED'] > 0) & (df['NUMBER OF PERSONS KILLED'] == 0),
    (df['NUMBER OF PERSONS KILLED'] > 0)
]

# Define corresponding values for each condition
values = ['No Injury', 'Injury', 'Fatal']

# Create the new column based on conditions
df['Accident Severity Category'] = np.select(conditions, values, default='Others')

# Display the updated DataFrame
print(df[['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 'Accident Severity Category']])

# %%
sns.countplot(data=df, x='Accident Severity Category', hue='Year',
              order=df['Accident Severity Category'].value_counts().index)

plt.title('Number of Each Category of Accident Severity by Year')
plt.xlabel('Accident Severity Category')
plt.ylabel('Count')
plt.legend(title='Year')

plt.show()
# %%
severity_counts = df.groupby(['Year', 'Accident Severity Category']).size().reset_index(name='Count')

# Create a separate pie chart for each year
years = severity_counts['Year'].unique()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

for i, year in enumerate(years):
    ax = axes[i // 3, i % 3]
    data = severity_counts[severity_counts['Year'] == year]
    labels = data['Accident Severity Category']
    sizes = data['Count']
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Year {year}')

plt.tight_layout()
plt.show()
# %%
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
severity_counts = df.groupby(['Year', 'Accident Severity Category']).size().reset_index(name='Count')

injury_data = severity_counts[severity_counts['Accident Severity Category'].isin(['Injury', 'No Injury'])]

fig, ax = plt.subplots(figsize=(10, 6))
sns.set_palette("pastel")

def update(frame):
    ax.clear()
    data = injury_data[injury_data['Year'] <= frame]
    sns.barplot(data=data, x='Year', y='Count', hue='Accident Severity Category', ax=ax)
    ax.set_title(f'Accident Severity Distribution ({frame})')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.legend(title='Accident Severity Category')

years = sorted(injury_data['Year'].unique())
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
by_factor_severity = df.groupby(['CONTRIBUTING FACTOR VEHICLE 1','Accident Severity Category']).count()
by_factor_severity.head(35)
by_factor_severity.reset_index(inplace=True)
# %%
total_counts = by_factor_severity.groupby('CONTRIBUTING FACTOR VEHICLE 1')['CRASH TIME'].sum()

#Total occurences
fatal_proportions = (
    by_factor_severity[by_factor_severity['Accident Severity Category'] == 'Injury']
    .groupby('CONTRIBUTING FACTOR VEHICLE 1')['CRASH TIME']
    .sum()
    / total_counts
).fillna(0)  

sorted_factors = fatal_proportions.sort_values(ascending=False)

top_10_factors = sorted_factors.head(10)
print(top_10_factors)
# %%

total_counts = by_factor_severity.groupby('CONTRIBUTING FACTOR VEHICLE 1')['CRASH TIME'].sum()

#Total occurences
fatal_proportions = (
    by_factor_severity[by_factor_severity['Accident Severity Category'] == 'Fatal']
    .groupby('CONTRIBUTING FACTOR VEHICLE 1')['CRASH TIME']
    .sum()
    / total_counts
).fillna(0)  

sorted_factors = fatal_proportions.sort_values(ascending=False)

top_10_factors = sorted_factors.tail(10)
print(top_10_factors)
# %%
