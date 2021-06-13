import pickle
import pandas as pd

with open('model.pickle','rb') as handle:
    naive_bayes = pickle.load(handle)
with open('vectorizer.pickle','rb') as handle:
    count_vector = pickle.load(handle)

test_review=[]
test_review.append(input())
review = pd.DataFrame(test_review)
review = count_vector.transform(review.stack())
predictions = naive_bayes.predict(review)

if predictions[0]==1:
    print("Scam")
else:
    print("Not scam")