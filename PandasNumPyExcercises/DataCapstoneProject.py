# imports
import pandas as pd
import seaborn as sns

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
df['reason'] = df['title'].apply(lambda x: x.split(':')[0])
print(df['reason'].value_counts())
sns.countplot(x='reason', data=df)
import matplotlib.pyplot as plt
plt.show()

