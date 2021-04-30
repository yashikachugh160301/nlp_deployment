import sys
import nltk
import sklearn
import pandas
import numpy
import pickle
import pandas as pd
import numpy as np

#loading the data
df= pd.read_table('/Users/yashikachugh/Desktop/nlp/SMSSpamCollection', header= None, encoding='utf-8')
classes=df[0]
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
Y=encoder.fit_transform(classes)
text=df[1]
#replacing regular expressions for email,url,phone number

processed=text.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailadd')

processed=processed.str.replace(r'^http\://\[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webadd')
processed=processed.str.replace(r'£|\$','moneysmb')
processed=processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenum')
processed=processed.str.replace(r'\d+(\.\d+)?','numbr')#remove punctuation
processed=processed.str.replace(r'[^\w\d\s]',' ')

#remove whitspace b/w words
processed=processed.str.replace(r'\s+',' ')

#remove leading and trailing
processed=processed.str.replace(r'^\s+|\s+?$',' ')

#lowercase
processed=processed.str.lower()
#remove stopwords
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
processed=processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

#stemming
ps=nltk.PorterStemmer()
processed=processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(processed).toarray()

pickle.dump(cv,open('transform.pkl','wb'))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()

classifier.fit(X_train, y_train)
filename='nlp_model.pkl'
pickle.dump(classifier,open(filename,'wb'))

