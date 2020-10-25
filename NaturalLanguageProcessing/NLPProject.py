# imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
yelp = pd.read_csv('C:/Users/user/Downloads/yelp.csv')
# chceck data
print(yelp.describe())
print(yelp.info())
print(yelp.head())
# length of text
yelp['text length'] = yelp['text'].apply(len)
print(yelp.head())
# exploratory data analysis
g = sns.FacetGrid(yelp, col="stars")
g.map(plt.hist, "text length", alpha=0.5)
plt.show()
sns.boxplot(x='stars', y='text length', data=yelp)
plt.show()
sns.countplot(x='stars', data=yelp)
plt.show()
# groupby
yelp.groupby(by='stars').mean()
df = yelp.groupby(by='stars').mean().corr()
print(df)
sns.heatmap(data=df, cmap='coolwarm', annot=True)
plt.show()
# NLP
yelp_class = yelp[(yelp['stars']==1) |(yelp['stars']==5)]
yelp_class['stars'].value_counts()
yelp_class.head()
X = yelp_class['text']
y = yelp_class['stars']
from sklearn.feature_extraction.text import CountVectorizer
# Convert a collection of text documents to a matrix of token counts
cv = CountVectorizer()
X = cv.fit_transform(X)
# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=42)
# training a model
#The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
# evaluation
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
#text processing
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)
#evaluation
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))