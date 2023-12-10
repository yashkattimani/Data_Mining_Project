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

rush_hour_start = 0
rush_hour_end = 24

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
hourly_data = df.groupby('Hour').apply(lambda x: x.loc[x['NUMBER OF PERSONS KILLED'].idxmax()])[['CONTRIBUTING FACTOR VEHICLE 1', 'NUMBER OF PERSONS KILLED']]

plt.figure(figsize=(12, 6))
sns.barplot(data=hourly_data, x=hourly_data.index, y='NUMBER OF PERSONS KILLED',hue='CONTRIBUTING FACTOR VEHICLE 1')
plt.title("Number of Persons Killed by Hour with Leading Contributing Factor")
plt.xlabel("Hour")
plt.ylabel("Number of Persons Killed")
plt.xticks(rotation=45)
plt.show()
# %%
hour_factor_data=pd.DataFrame(df.groupby(['Hour','CONTRIBUTING FACTOR VEHICLE 1'])['NUMBER OF PERSONS KILLED'].sum())
hour_factor_data.head(20)

# %%
hour_factor_data.reset_index(inplace=True)
hour_factor_data.set_index(['Hour'],inplace=True)
hour_factor_data.head()
# %%
max_factors = hour_factor_data.groupby('Hour').apply(lambda x: x[x['NUMBER OF PERSONS KILLED'] == x['NUMBER OF PERSONS KILLED'].max()])

# Create a bar graph
plt.figure(figsize=(10, 6))
plt.bar(max_factors.index, max_factors['NUMBER OF PERSONS KILLED'])
plt.xlabel('Hour')
plt.ylabel('Max Number of Persons Killed')
plt.title('Max Number of Persons Killed and Corresponding Factor by Hour')
plt.xticks(rotation=45)
plt.show()

# %%
df.head()
# %%
by_vehicle=pd.DataFrame(df.groupby(['VEHICLE TYPE CODE 1','VEHICLE TYPE CODE 2'])['NUMBER OF PERSONS KILLED'].sum())
by_vehicle=by_vehicle[by_vehicle['NUMBER OF PERSONS KILLED']>10]
print(by_vehicle.shape)
by_vehicle.head()
# %%
by_vehicle.reset_index(inplace=True)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=by_vehicle,x='VEHICLE TYPE CODE 1',y='VEHICLE TYPE CODE 2',hue='NUMBER OF PERSONS KILLED',size='NUMBER OF PERSONS KILLED')
plt.xlabel('Vehicle 1')
plt.ylabel('Vehicle 2')
plt.title('Vehicle 1 vs Vehicle 2 ')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(data=by_vehicle,x='VEHICLE TYPE CODE 1',y='NUMBER OF PERSONS KILLED',hue='VEHICLE TYPE CODE 2',s=100)
plt.xlabel('Vehicle 1')
plt.ylabel('Persons Killed')
plt.title('Vehicle 1 vs Persons killed')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()
# %%

cyclist_accidents=rush_hour_data[rush_hour_data['NUMBER OF CYCLIST INJURED']>0]
top_10_cyclist_locations = cyclist_accidents.sort_values(by='NUMBER OF CYCLIST INJURED', ascending=False).head(10)

m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)

locations = top_10_cyclist_locations[['LATITUDE', 'LONGITUDE']].values

gradient = {
    0.2: 'blue',
    0.4: 'green',
    0.6: 'orange',
    1.0: 'red'
}

HeatMap(data=locations, gradient=gradient, radius=15).add_to(m)

m
# %%
df.head()
# %%
df.columns
# %%
by_borough = df.groupby(['BOROUGH','Year'])[['NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED','NUMBER OF PEDESTRIANS INJURED',
       'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF CYCLIST INJURED',
       'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST INJURED',
       'NUMBER OF MOTORIST KILLED']].sum()
by_borough.head()
#%% 
by_borough.reset_index(inplace=True)
# %%
plt.figure(figsize=(10, 8))
sns.scatterplot(data=by_borough, x='Year', y='NUMBER OF PERSONS INJURED', hue='BOROUGH',size='NUMBER OF PERSONS KILLED')
plt.xlabel('Year')
plt.ylabel('Injured')
plt.title("Scatter plot of People Injured by Year")
plt.show()
# %%
import plotly.express as px

fig = px.scatter(by_borough, y='NUMBER OF PERSONS KILLED', x='NUMBER OF PERSONS INJURED', color='BOROUGH', size='NUMBER OF PERSONS KILLED', animation_frame='Year')

fig.update_xaxes(range=[0, 17000]) 
fig.update_yaxes(range=[0, 100])   

fig.show()

# %%
df.columns
# %%

fig = px.scatter(by_borough, y='NUMBER OF CYCLIST KILLED', x='NUMBER OF MOTORIST KILLED', color='BOROUGH', size='NUMBER OF CYCLIST KILLED', animation_frame='Year')

fig.update_xaxes(range=[0, 50]) 
fig.update_yaxes(range=[0, 20])   

fig.show()

# %%
df.columns

#%%
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df,x='CONTRIBUTING FACTOR VEHICLE 1',y='NUMBER OF PERSONS INJURED')
plt.xlabel('Ye')
plt.ylabel('Injured')
plt.title("Scatter plot of People Injured by Year")
plt.show()
#%%
plt.figure(figsize=(10, 8))
sns.stripplot(data=df,x='CONTRIBUTING FACTOR VEHICLE 1',y='NUMBER OF PERSONS INJURED',jitter=True)
plt.xticks(rotation=45)
plt.show()
# %%
by_factor_year = pd.DataFrame(df.groupby(['CONTRIBUTING FACTOR VEHICLE 1','Year'])[['NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED']].sum())
# %%
by_factor_year.head()
by_factor_year.reset_index(inplace=True)
# %%

fig = px.scatter(by_factor_year, y='NUMBER OF PERSONS KILLED',x='NUMBER OF PERSONS INJURED' ,color='CONTRIBUTING FACTOR VEHICLE 1', animation_frame='Year')

fig.update_xaxes(range=[0, 170]) 
fig.update_yaxes(range=[0, 10])   

fig.show()


# %%
df['CONTRIBUTING FACTOR VEHICLE 1'].unique()
# %%
counts=df.groupby('CONTRIBUTING FACTOR VEHICLE 1').count()
# %%
counts.reset_index(inplace=True)
#%%
counts
# %%
top_10_factors = counts.sort_values(by='Year', ascending=False).head(11)

print("Top 10 Contributing Factors by Counts:")
print(top_10_factors)

# Now, create a bar graph to visualize the top 10 contributing factors
plt.figure(figsize=(10, 6))
plt.bar(top_10_factors['CONTRIBUTING FACTOR VEHICLE 1'], top_10_factors['Year'])
plt.title('Top 10 Contributing Factors by Counts')
plt.xlabel('Contributing Factors')
plt.ylabel('Counts')
plt.xticks(rotation=45)
plt.show()
# %%
top_factors = top_10_factors['CONTRIBUTING FACTOR VEHICLE 1'].tolist()
filtered_df = df[df['CONTRIBUTING FACTOR VEHICLE 1'].isin(top_factors)]

# Group the filtered DataFrame by year and factor, counting incidents
grouped_df = filtered_df.groupby(['Year', 'CONTRIBUTING FACTOR VEHICLE 1']).size().reset_index(name='Incidents')

plt.figure(figsize=(12, 6))
sns.lineplot(data=grouped_df, x='Year', y='Incidents', hue='CONTRIBUTING FACTOR VEHICLE 1')
plt.title('Yearly Incidents by Contributing Factor')
plt.xlabel('Year')
plt.ylabel('Number of Incidents')
plt.legend(title='Contributing Factor', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()


# %%
df.head()

# %%
df['Month'] = df['CRASH DATE'].dt.month
df['Day'] = df['CRASH DATE'].dt.day
df['Day of Week'] = df['CRASH DATE'].dt.strftime('%A')
# %%
by_month=df.groupby(['Month','Year']).count()
by_day_of_week=df.groupby('Day of Week').count()

by_month.reset_index(inplace=True)
# %%
plt.figure(figsize=(12, 6))
sns.lineplot(data=by_month, x='Month', y='ZIP CODE',hue='Year')
plt.title('Monthly Incidents')
plt.xlabel('Month')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.show()

# %%

df.columns
# %%
df['BOROUGH'].unique()
# %%
unique_df = pd.DataFrame()

for column in df.columns:
    unique_df[column] = pd.Series(df[column].unique())

# Display the new DataFrame
print(unique_df)
# %%
unique_df[['BOROUGH', 'ZIP CODE', 'LATITUDE',
       'LONGITUDE', 'LOCATION', 'ON STREET NAME', 'CROSS STREET NAME',
       'OFF STREET NAME', 'NUMBER OF PERSONS INJURED',
       'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS INJURED',
       'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF CYCLIST INJURED',
       'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST INJURED',
       'NUMBER OF MOTORIST KILLED', 'CONTRIBUTING FACTOR VEHICLE 1',
       'CONTRIBUTING FACTOR VEHICLE 2', 'CONTRIBUTING FACTOR VEHICLE 3',
       'CONTRIBUTING FACTOR VEHICLE 4', 'CONTRIBUTING FACTOR VEHICLE 5',
       'COLLISION_ID', 'VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2',
       'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5']].head(20)
# %%
df['CONTRIBUTING FACTOR VEHICLE 1'].unique()
# %%
df.dtypes
# %%
plt.figure(figsize=(20,8))
sns.histplot(data = df,x='CONTRIBUTING FACTOR VEHICLE 1')
plt.xticks(rotation=45)
plt.show()

# %%
