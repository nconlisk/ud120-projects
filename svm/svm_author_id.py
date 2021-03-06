#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
from sklearn.svm import SVC

clf = SVC(kernel="linear", gamma=1.0)

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


