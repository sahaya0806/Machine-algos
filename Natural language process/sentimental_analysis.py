
# 1] Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2]Importing the dataset in the tsv form
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# 3]Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  # 4] Removing the not from stopwords
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
print(" The deep cleaned corpus of reviews")
print(corpus)

# 5] Bag of Words model(Tokenisation)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1300)
# 6] Converting X to 2d array(Sparse Matrix)
X = cv.fit_transform(corpus).toarray() 
y = dataset.iloc[:, -1].values

# 6] Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# 7]Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# 8]Predicting the Test set results
y_pred = classifier.predict(X_test)
print("Comparing the y_pred and y_test")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# 9]Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print("Accuracy score and Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# 10]Predicting Sentimental Analysis for new_reveiw
new_review = " I Love this restaurent so much"
new_review = re.sub('[^a-zA-Z]', ' ',new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords= stopwords.words('english')
all_stopwords.remove('not')
new_review=[ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)

# 11] Printnig the predicted test result
print(" The result for the new review")
print(new_y_pred)

