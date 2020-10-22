import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
customers = pd.read_csv("C:/Users/user/Downloads/Ecommerce Customers.csv")
#check database
print(customers.head())
print(customers.info())
print(customers.describe())
#visualization
j = sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
j.annotate(stats.pearsonr)
plt.show()
j = sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
j.annotate(stats.pearsonr)
plt.show()
j = sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind='hex')
j.annotate(stats.pearsonr)
plt.show()
sns.pairplot(data=customers)
plt.show()
#linear model
sns.lmplot(y='Yearly Amount Spent',x='Length of Membership', data=customers)
#training and test data
from sklearn.model_selection import train_test_split

X=customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
Y=customers['Yearly Amount Spent']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3, random_state=101)

#training model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, Y_train)
#coefficient of model
print(lm.coef_)
#predicting test data
predicts = lm.predict(X_test)
plt.scatter(Y_test,predicts)
#evaluating model
from sklearn import metrics
ma = metrics.mean_absolute_error(Y_test,predicts)
ms = metrics.mean_squared_error(Y_test,predicts)
rms = np.sqrt(ms)
print(ma,ms,rms)
#residuals
sns.distplot((Y_test-predicts),bins=50)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)