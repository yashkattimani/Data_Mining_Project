#%%


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
