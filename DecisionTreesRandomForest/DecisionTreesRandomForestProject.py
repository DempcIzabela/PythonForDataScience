# import library
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
# import data
loans = pd.read_csv('C:/Users/user/Downloads/loan_data.csv')
#checking data
print(loans.info())
print(loans.describe())
print(loans.head())
# exploratory data analysis
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist( bins=30, label='Credit.Policy=1', alpha=0.4)
loans[loans['credit.policy']==0]['fico'].hist( bins=30, label='Credit.Policy=0', alpha=0.4)
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist( bins=30, label='not.fully.paid=1', alpha=0.4)
loans[loans['not.fully.paid']==0]['fico'].hist( bins=30, label='not.fully.paid=0', alpha=0.4)
plt.legend()
plt.show()

plt.figure(figsize=(10,4))
sb.countplot(x='purpose',data=loans,hue='not.fully.paid')
plt.show()

sb.jointplot(data=loans, x='fico', y='int.rate',xlim=(600,850), ylim=(0.00,0.25))
plt.show()

sb.lmplot(x='fico', y='int.rate', hue='credit.policy', col='not.fully.paid', data=loans)
plt.show()

#dummy variables
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
print(final_data.head())

#train test split
from sklearn.model_selection import train_test_split
y=final_data['not.fully.paid']
X=final_data.drop('not.fully.paid', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#decision tree model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X=X_train,y=y_train)
predicts = dtree.predict(X_test)

#evaluation
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(predicts, y_test)
print(classification_report(predicts, y_test))

#training the random forest model
from sklearn.ensemble import RandomForestClassifier
rforest = RandomForestClassifier(n_estimators=600)
rforest.fit(X=X_train, y=y_train)
predicts = rforest.predict(X_test)

#evaluation
print(classification_report(predicts,y_test))
print(confusion_matrix(y_test,predicts))