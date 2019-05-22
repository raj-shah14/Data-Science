import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('1-restaurant-train.csv',delimiter='\t',quoting=3) #Quoting to remove quotes
data.columns = ['Liked','Review']
import re
import nltk

for i,j in enumerate(data['Liked']):
    data['Liked'][i]=re.sub('[^0-9]','',j)
    if(data['Liked'][i]== '4' or data['Liked'][i]=='5'):
        data['Liked'][i]=1
    elif(data['Liked'][i]== '1' or data['Liked'][i]=='2' or data['Liked'][i]=='3'):
        data['Liked'][i]=0



from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from autocorrect import spell

clean_corpus=[]
for i in data['Review']:
    review=re.sub('[^A-Za-z]',' ',i)
    review=review.lower()
    review=review.split(' ') 
#
#    for j in review:
#        if(spell(j) != j):
#            loc=review.index(j)
#            review[loc]=spell(j)
            
#    review=[x.lower() for x in review]
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    clean_corpus.append(review)
    

#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
countvec=CountVectorizer(max_features=500)
X=countvec.fit_transform(clean_corpus).toarray()
Y=data.iloc[:,0].values

#Splitting into test and train
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
ev=pca.explained_variance_ratio_


#naive-bayes Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
classifier =GaussianNB()  #---> Accuracy = 73%
#classifier=tree.DecisionTreeClassifier(random_state=0) #--> Accuracy = 68%
#classifier=SGDClassifier() # --> Accuracy =75%
#classifier=RandomForestClassifier(random_state=0) # --> ACcuracy = 67%
y_train=y_train.astype('int')
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
y_test=y_test.astype('int')
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)    