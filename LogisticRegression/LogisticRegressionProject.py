import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
ad_data = pd.read_csv("C:/Users/user/Downloads/advertising.csv")
#chcecking dataframe
print(ad_data.head())
print(ad_data.info())
print(ad_data.describe())
#exploratory data analysis
sns.distplot(a=ad_data['Age'], kde=False, bins=30)
plt.show()
sns.jointplot(x='Age', y='Area Income', data=ad_data)
plt.show()
sns.jointplot(x='Age', y='Daily Time Spent on Site',data=ad_data, kind='kde', color='r')
plt.show()
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage',data=ad_data, color='g')
plt.show()
sns.pairplot(ad_data)
plt.show()
#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']],
                                                    ad_data['Clicked on Ad'], test_size=0.30)
#building model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predict = logmodel.predict(X_test)
#evalutaion
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))