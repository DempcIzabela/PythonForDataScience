# imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/user/Downloads/911.csv")
# information
df.info()
# 5 first rows
print(df.head())
# 5 most often zip-code
print(df['zip'].value_counts().head(5))
# number of unique tittle
print(df['title'].nunique())
# new column reason
df['reason'] = df['title'].apply(lambda title_name: title_name.split(':')[0])
print(df['reason'].value_counts())
sns.countplot(x='reason', data=df)

plt.show()
# new column, hour, month, day of week
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['hour'] = df['timeStamp'].apply(lambda hour_name: hour_name.hour)
df['month'] = df['timeStamp'].apply(lambda month_name: month_name.month)
df['day of week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
print(df.head())
# map
dictionary_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['day of week'] = df['day of week'].map(dictionary_map)
# plot
sns.countplot(x='day of week', hue='reason', data=df)
plt.show()
sns.countplot(x='month', hue='reason', data=df)
plt.show()
# byMonth
byMonth = df.groupby(df['month']).count()
x = pd.DataFrame(df['month'].value_counts().sort_index())
print(x)
x['index'] = x.index
sns.lmplot(x='index', y='month', data=x)
plt.show()
# new column date
df['Date'] = df["timeStamp"].apply(lambda data_name: data_name.date())
df['Date'].value_counts().plot()
# new column taffic
traffic = df[df['reason'] == 'Traffic']
traffic['Date'].value_counts().plot()
# heatmap
dayHour = df.groupby(by=['day of week', 'hour']).count()['reason'].unstack()
dayHour.head()
sns.heatmap(dayHour, cmap="coolwarm")
plt.show()
sns.clustermap(dayHour)
plt.show()
df.groupby(by=['day of week', 'month']).count()['reason'].unstack()
