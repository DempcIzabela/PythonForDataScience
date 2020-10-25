# import libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# import database
knn = pd.read_csv("C:/Users/user/Downloads/KNN_Project_Data.csv")
# exploratory analysis
sns.pairplot(data=knn, hue='TARGET CLASS')
plt.show()
# standarize the variables
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(knn.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(knn.drop('TARGET CLASS', axis=1))
knn_feat = pd.DataFrame(data=scaled_features, columns=knn.columns[:-1])
print(knn_feat.head())

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, knn['TARGET CLASS'], test_size=0.30)
# KNN

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)
predict = knn_model.predict(X_test)

# evaluation
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))

# for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list
predict(np.mean(predict != y_test))
error_rate = []
for k in range(1, 40):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    pred_k = knn_model.predict(X_test)
    error_rate.append(np.mean(pred_k != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train, y_train)
predict = knn_model.predict(X_test)
print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
