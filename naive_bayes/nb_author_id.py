#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess_v3 import preprocess
sys.path.remove("../tools/")
sys.path.append("../choose_your_own")
from class_vis import prettyPicture

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

training_time_start = time()
clf.fit(features_train, labels_train)
print("training time: ", round(time()-training_time_start,3), "s")

predict_time_start = time()
pred = clf.predict(features_test)
print("prediction time: ", round(time()-predict_time_start,3), "s")

accuracy = clf.score(features_test, labels_test)

print("prediction accuracy: ", accuracy)

prettyPicture(clf, features_test, labels_test)
#########################################################


