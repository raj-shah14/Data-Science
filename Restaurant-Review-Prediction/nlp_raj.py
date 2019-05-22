import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3) #Quoting to remove quotes

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from autocorrect import spell

clean_corpus=[]
for i in data['Review']:
    review=re.sub('[^a-zA-Z]',' ',i)
    review=review.lower()
    review=review.split()
    
    #Checks for Spellings
    for j in review:
        if(spell(j) != j):
            loc=review.index(j)
            review[loc]=spell(j)
    
    review=[x.lower() for x in review]
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    clean_corpus.append(review)
#print(clean_corpus)

#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
countvec=CountVectorizer(max_features=1500)
X=countvec.fit_transform(clean_corpus).toarray()
Y=data.iloc[:,1].values

#Splitting into test and train
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


#naive-bayes Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
#classifier =GaussianNB()  #---> Accuracy = 73%
#classifier=tree.DecisionTreeClassifier(random_state=0) #--> Accuracy = 68%
classifier=SGDClassifier() # --> Accuracy =75%
#classifier=RandomForestClassifier(random_state=0) # --> ACcuracy = 67%
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)    