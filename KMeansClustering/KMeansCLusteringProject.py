#imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
college = pd.read_csv("C:/Users/user/Downloads/College_Data.csv", index_col=0)
#check data
print(college.head())
print(college.info())
#exploratory data analysis
sns.set_style('whitegrid')
sns.lmplot(x='Room.Board', y='Grad.Rate', data=college, hue='Private', height=6, palette='coolwarm')
plt.show()

sns.set_style('whitegrid')
sns.lmplot( x='Outstate', y='F.Undergrad', data=college, hue='Private', size=6)
plt.show()

g = sns.FacetGrid(college, hue='Private',height=4, palette='coolwarm',aspect=2)
g.map(plt.hist, "Outstate", alpha=0.5)
plt.show()

g = sns.FacetGrid(college, hue='Private',height=4, palette='coolwarm',aspect=2)
g.map(plt.hist, "Grad.Rate", alpha=0.5)
plt.show()

date = college[college['Grad.Rate']<=100]
g = sns.FacetGrid(date, hue='Private',height=4, palette='coolwarm',aspect=2)
g.map(plt.hist, "Grad.Rate", alpha=0.5)
plt.show()

# k means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(college.drop('Private',axis=1))
print(kmeans.cluster_centers_)

#model evaluation
def f(row):
    if row['Private'] == 'Yes':
        val = 1
    else:
        val = 0
    return val

college['Cluster'] = college.apply(f, axis=1)
print(college.head())

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(college['Cluster'],kmeans.labels_))
print(classification_report(college['Cluster'],kmeans.labels_))