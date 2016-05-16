#!/usr/bin/python
#coding:utf-8
'''
'''
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import pandas as pd

from time import time
import unicodedata
import numpy as np
import glob
import json as js
import scipy
import os
import csv
import numpy as np
import operator
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.utils.extmath import density

testFile = r"/Users/Jason/Documents/deeplearningNLP/dataset/CleanDataset/test/"
trainFile = r"/Users/Jason/Documents/deeplearningNLP/dataset/CleanDataset/train/"

#allFiles = glob.glob(fileDir + "/*.csv")

def getDataFrame(fileDir):
    allFiles = glob.glob(fileDir + "/*.csv")
    frame = pd.DataFrame()
    column = ['section', 'title', 'abstract']
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None, names=column)
        list_.append(df)
    frame = pd.concat(list_)
    return frame


# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time



trainDf = getDataFrame(trainFile)
testDf = getDataFrame(testFile)


y_train = trainDf['section'].tolist()
train_messages_list = trainDf['abstract'].tolist()

y_test = testDf['section'].tolist()
test_messages_list = testDf['abstract'].tolist()
#train_messages_list, train_label = get_train_file(train_set)

train_messages_list = map(str, train_messages_list)#convert Unicode to utf-8
test_messages_list = map(str, test_messages_list)
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(train_messages_list)
print X_train.shape
#print X_train, X_train.shape, len(train_label)

X_test = vectorizer.transform(test_messages_list)
print X_test.shape
# clf = svm.LinearSVC()
# clf.fit(X_train, train_label)
#
# print clf.score(X_test, test_label)


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive")):
    print('=' * 80)
    print(name)
    benchmark(clf)

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    benchmark(LinearSVC(loss='l2', penalty=penalty,dual=False, tol=1e-3))

    # Train SGD model
    benchmark(SGDClassifier(alpha=.0001, n_iter=50,penalty=penalty))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
benchmark(MultinomialNB(alpha=.01))

print('=' * 80)
print("BernoulliNB")
benchmark(BernoulliNB(alpha=.01))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.




