#%%[markdown]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
df = pd.read_csv('C:/Users/nupur/computer/Desktop/Data Mining/Motor_Vehicle_Collisions_-_Crashes.csv')
df.head()
# %%
#Cleaning the NA or empty values in the Borough column.
df_clean = df.dropna(subset=['BOROUGH'])


# %%
import plotly.graph_objects as go
from plotly.offline import iplot

# Calculate the sums for each borough
sums_df = df.groupby('BOROUGH', as_index=False).sum()

# Add a total column
sums_df['TOTAL'] = sums_df['NUMBER OF CYCLIST KILLED'] + sums_df['NUMBER OF MOTORIST KILLED'] + sums_df['NUMBER OF PEDESTRIANS KILLED']

# Calculate percentages
sums_df['CYCLIST_KILLED_PERCENT'] = (sums_df['NUMBER OF CYCLIST KILLED'] / sums_df['TOTAL']) * 100
sums_df['MOTORIST_KILLED_PERCENT'] = (sums_df['NUMBER OF MOTORIST KILLED'] / sums_df['TOTAL']) * 100
sums_df['PEDESTRIANS_KILLED_PERCENT'] = (sums_df['NUMBER OF PEDESTRIANS KILLED'] / sums_df['TOTAL']) * 100

# Create traces for each category with percentages
trace1 = go.Bar(
    x=sums_df['BOROUGH'],
    y=sums_df['CYCLIST_KILLED_PERCENT'],
    name='Cyclist Killed',
    marker=dict(color='blue')
)

trace2 = go.Bar(
    x=sums_df['BOROUGH'],
    y=sums_df['MOTORIST_KILLED_PERCENT'],
    name='Motorist Killed',
    marker=dict(color='orange')
)

trace3 = go.Bar(
    x=sums_df['BOROUGH'],
    y=sums_df['PEDESTRIANS_KILLED_PERCENT'],
    name='Pedestrians Killed',
    marker=dict(color='red')
)

data = [trace1, trace2, trace3]

layout = go.Layout(
    title='Average Percentage of Deaths by Borough',
    xaxis=dict(title='Borough'),
    yaxis=dict(title='Percentage of Deaths', tickformat='.2f'),
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

# %%
import plotly.express as px
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
df['YEAR'] = df['CRASH DATE'].dt.year

# Group by 'YEAR' and 'BOROUGH' and calculate the average number of persons injured
# Since we're using mock data, we'll just count the number of incidents per year for simplicity
df_average = df.groupby(['YEAR', 'BOROUGH']).agg({'NUMBER OF PERSONS INJURED': 'mean'}).reset_index()

# Create the line chart
fig = px.line(df_average, x='YEAR', y='NUMBER OF PERSONS INJURED', color='BOROUGH',
              labels={'PERSONS_INJURED': 'Average Number of Persons Injured'},
              title='Average Number of Persons Injured by Year and Borough')

fig.update_layout(
    width=600,  # You can adjust this value as needed
)
# Update the x-axis to show every year
fig.update_xaxes(dtick=1)

# Show the plot
fig.show()
# %%
# %%
import plotly.express as px
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
df['YEAR'] = df['CRASH DATE'].dt.year

# Group by 'YEAR' and 'BOROUGH' and calculate the average number of persons injured
# Since we're using mock data, we'll just count the number of incidents per year for simplicity
df_average = df.groupby(['YEAR', 'BOROUGH']).agg({'NUMBER OF PERSONS KILLED': 'mean'}).reset_index()

# Create the line chart
fig = px.line(df_average, x='YEAR', y='NUMBER OF PERSONS KILLED', color='BOROUGH',
              labels={'PERSONS_INJURED': 'Average Number of Persons Killed'},
              title='Average Number of Persons Killed by Year and Borough')

fig.update_layout(
    width=600,  # You can adjust this value as needed
)
# Update the x-axis to show every year
fig.update_xaxes(dtick=1)

# Show the plot
fig.show()
# %%
import pandas as pd
from scipy import stats

# Let's assume your DataFrame 'df' has 'CYCLIST_KILLED', 'PEDESTRIAN_KILLED', and 'MOTORIST_KILLED' columns.

# First, we calculate the average number of people killed in each category.
# This assumes that 'BOROUGH' and 'YEAR' are also columns in your DataFrame.
averages = df.groupby(['BOROUGH', 'YEAR']).agg({
    'NUMBER OF CYCLIST KILLED': 'mean',
    'NUMBER OF PEDESTRIANS KILLED': 'mean',
    'NUMBER OF MOTORIST KILLED': 'mean'
}).reset_index()

# For the Chi-Squared test, we would need observed counts, not means.
# However, if you have the observed frequencies, you can perform a Chi-Squared test as follows:
# chi2, p, dof, ex = stats.chi2_contingency(observed_frequencies)
# observed_frequencies should be a 2D array-like structure representing the contingency table.

# For the ANOVA test, you can use the f_oneway function from scipy.stats which performs a 1-way ANOVA.
f_value, p_value = stats.f_oneway(
    averages['NUMBER OF CYCLIST KILLED'],
    averages['NUMBER OF PEDESTRIANS KILLED'],
    averages['NUMBER OF MOTORIST KILLED']
)

# %%
print(f_value)
# %%
print(p_value)
# %%
