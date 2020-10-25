#import data from seaborn library
import seaborn as sns
iris = sns.load_dataset('iris')
#imports
import matplotlib.pyplot as plt
#exploratory data analysis
sns.pairplot(data = iris, hue='species',palette='Dark2')
plt.show()
zbior = iris[iris['species'] == 'setosa'][['sepal_length','sepal_width']]
print(zbior.head())
sns.kdeplot(zbior['sepal_width'],zbior['sepal_length'],
            shade=True, shade_lowest=False, cmap='plasma')
plt.show()
#train test split
from sklearn.model_selection import train_test_split
X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
predicts = model.predict(X_test)
#model evaluation
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,predicts))
print(classification_report(y_test, predicts))
#gridsearch practise
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train, y_train)
