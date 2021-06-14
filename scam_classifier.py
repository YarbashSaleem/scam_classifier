import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection  import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#Data Preprocessing
review_database = open("review_database.csv","r")
df=pd.read_table('review_database.csv',names=['label','review'],delimiter=',')
df['label']=df.label.map({'ham ':0, 'scam':1})

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], random_state=1,test_size=.2)



# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)
predictions = naive_bayes.predict(testing_data)

with open('model.pickle','wb') as handle:
    pickle.dump(naive_bayes,handle)
with open('vectorizer.pickle','wb') as handle:
    pickle.dump(count_vector,handle)


print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions)))
print('Recall score: ', format(recall_score(y_test,predictions)))
print('F1 score: ', format(f1_score(y_test,predictions)))
