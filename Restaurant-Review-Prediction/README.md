# Restaurant-Review-Prediction-using-Bag-of-Words
 Using BOW model on Restaurant Review dataset to predict the accuracy and comparing with different classifiers.
 Training set had 1000 reviews. In preprocessing step, data had to be cleaned first. 
 
 Classifier Type | Accuracy
 ---| --- | 
 SGD | 75%
 Naive-Bayes | 73%
 Decision Tree | 68%
 Random Forest | 67%

I downloaded the Kaggle Restaurant review set from [here](https://www.kaggle.com/c/restaurant-reviews/data)
It contains 80000 reviews in training set and 30000 in test set.
I trained a NaiveBayes classifier an obtained an accuracy of 73% using Bag of Words Model.

Used PCA to reduce the dimensionality of the vector.
Confusion Matrix

Neagtive-0 | Positive-1
--- | --- |
919 | 4917
966 | 9611

