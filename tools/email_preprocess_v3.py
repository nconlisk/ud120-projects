#!/usr/bin/python



# Initialisation and function code updated for python 3.7.1 and scikit-learn 0.20.1

import pickle
import numpy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif



def preprocess(words_file = "../tools/word_data.pkl", authors_file="../tools/email_authors.pkl"):
    """ 
        this function takes a pre-made list of email texts (by default word_data.pkl)
        and the corresponding authors (by default email_authors.pkl) and performs
        a number of preprocessing steps:
            -- splits into training/testing sets (10% testing)
            -- vectorizes into tfidf matrix
            -- selects/keeps most helpful features

        after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

        4 objects are returned:
            -- training/testing features
            -- training/testing labels

    """

    ### the words (features) and authors (labels), already largely preprocessed
    ### this preprocessing will be repeated in the text learning mini-project
    authors_file_handler = open(authors_file, "r")
    afh_eol_removed = [line.rstrip("\r\n") for line in authors_file_handler.readlines()]
    afh_eol_corrected = "\n".join(afh_eol_removed)
    afh_unix = afh_eol_corrected.encode("utf-8")
    authors = pickle.loads(afh_unix)
    authors_file_handler.close()
   

    words_file_handler = open(words_file, "r")
    wfh_eol_removed = [line.rstrip("\r\n") for line in words_file_handler.readlines()]
    wfh_eol_corrected = "\n".join(wfh_eol_removed)  #data fine to here
    wfh_unix = wfh_eol_corrected.encode("utf-8")
    word_data = pickle.loads(wfh_unix) #without encoding this step fails
    words_file_handler.close()    
    
    
    # Could replace this section with a function and feed it authors/words to reduce code repetition.

    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)



    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)



    ### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    ### info on the data
    print ("no. of Chris training emails:", sum(labels_train))
    print ("no. of Sara training emails:", len(labels_train)-sum(labels_train))
    
    return features_train_transformed, features_test_transformed, labels_train, labels_test
