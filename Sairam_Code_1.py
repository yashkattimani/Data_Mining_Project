#%%

import pandas as pd

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
# %%
df.head()
# %%
df['Year'].unique()
# %%
df['Year'].value_counts()
# %%
import matplotlib.pyplot as plt
import seaborn as sns

yearly_counts = df['Year'].value_counts().reset_index()
yearly_counts.columns = ['Year', 'Count']

plt.figure(figsize=(10, 6))
sns.barplot(data=yearly_counts, x='Year', y='Count')
plt.title('Counts of Data Points by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# %%

df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'])
df['Hour'] = df['CRASH TIME'].dt.hour

plt.figure(figsize=(12, 6))
sns.violinplot(data=df, y='Hour')
plt.title('Distribution of Crash Times by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.show()

# %%
df.head()


# %%

rush_hour_start = 7
rush_hour_end = 9

rush_hour_data = df[(df['Hour'] >= rush_hour_start) & (df['Hour'] <= rush_hour_end)]

location_counts = rush_hour_data['BOROUGH'].value_counts()

# Plot the accident hotspots
location_counts.plot(kind='bar')
plt.title("Accident Hotspots during Rush Hour")
plt.xlabel("Borough")
plt.ylabel("Number of Accidents")
plt.show()

# %%
import folium
from folium.plugins import HeatMap

rush_hour_data = df[(df['Hour'] >= rush_hour_start) & (df['Hour'] <= rush_hour_end)]

m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)

locations = rush_hour_data[['LATITUDE', 'LONGITUDE']].dropna()

# Define the color gradient for the heatmap
gradient = {
    0.2: 'blue',
    0.4: 'green',
    0.6: 'orange',
    1.0: 'red'
}

# Create a heatmap layer with the color scale
HeatMap(data=locations, gradient=gradient, radius=15).add_to(m)

# Display the map
m

# %%
import folium
from folium.plugins import HeatMap

# Filter data for accidents during rush hour
rush_hour_data = df[(df['Hour'] >= rush_hour_start) & (df['Hour'] <= rush_hour_end)]

# Group the data by latitude and longitude and count the number of accidents in each location
locations = rush_hour_data.groupby(['LATITUDE', 'LONGITUDE']).size().reset_index(name='ACCIDENT_COUNT')

# Create a map centered around a specific location (e.g., New York City)
m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)

# Create a heatmap layer using the aggregated accident count
HeatMap(data=locations[['LATITUDE', 'LONGITUDE', 'ACCIDENT_COUNT']], radius=15).add_to(m)

# Display the map
m

# %%
df.head()
# %%
df.columns


# %%
print(df['LATITUDE'].max())
print(df['LATITUDE'].min())
# %%
print(df['LONGITUDE'].max())
print(df['LONGITUDE'].min())

# %%
plt.figure(figsize=(8, 10))
sns.scatterplot(data=df, y='LONGITUDE', x='LATITUDE', hue='BOROUGH')

plt.xlim(40,41)
plt.ylim(-73,-75)


plt.show()
# %%
df.head()
#%%
by_hour = pd.DataFrame(df.groupby('Hour')[['NUMBER OF PERSONS INJURED',
       'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS INJURED',
       'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF CYCLIST INJURED',
       'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST INJURED',
       'NUMBER OF MOTORIST KILLED']].sum())

sns.barplot(data=by_hour,x='Hour',y='NUMBER OF CYCLIST KILLED', label='Cyclists Killed', color='b')


plt.title('Number of Cyclists vs People Killed by Hour')
plt.xlabel('Hour')
plt.ylabel('Casualities')
plt.xticks(rotation=0)
plt.show()

# %%
df.columns
# %%
by_hour
# %%

sns.lineplot(data=by_hour,x='Hour',y='NUMBER OF CYCLIST KILLED', label='Cyclists Killed', color='b')
sns.lineplot(data=by_hour,x='Hour',y='NUMBER OF PEDESTRIANS KILLED', label='Pedestrians Killed', color='r')
sns.lineplot(data=by_hour,x='Hour',y='NUMBER OF MOTORIST KILLED', label='Motorists Killed', color='g')
sns.lineplot(data=by_hour,x='Hour',y='NUMBER OF PERSONS KILLED', label='Total People Killed', color='y')





plt.title('Number of Cyclists vs People Killed by Hour')
plt.xlabel('Hour')
plt.ylabel('Casualities')
plt.xticks(rotation=0)
plt.show()
# %%
df.columns
# %%
df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts()
#%%
df['VEHICLE TYPE CODE 1'].value_counts()

# %%
plt.figure(figsize=(10,10))
sns.scatterplot(data=df,x='Hour',y='NUMBER OF PERSONS KILLED',hue='CONTRIBUTING FACTOR VEHICLE 1')
plt.xlabel('Hour')
plt.ylabel('Number of Casualities')
plt.show()

# %%
hourly_data = df.groupby('Hour').apply(lambda x: x.loc[x['NUMBER OF PERSONS KILLED'].idxmax()])[['CONTRIBUTING FACTOR VEHICLE 1', 'NUMBER OF PERSONS KILLED']]

plt.figure(figsize=(12, 6))
sns.barplot(data=hourly_data, x=hourly_data.index, y='NUMBER OF PERSONS KILLED',hue='CONTRIBUTING FACTOR VEHICLE 1')
plt.title("Number of Persons Killed by Hour with Leading Contributing Factor")
plt.xlabel("Hour")
plt.ylabel("Number of Persons Killed")
plt.xticks(rotation=45)
plt.show()
# %%
