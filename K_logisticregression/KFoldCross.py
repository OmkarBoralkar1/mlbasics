# English:
#
# k-Fold Cross-Validation is a technique used to assess the performance of a machine learning model.
# The dataset is divided into k subsets, and the model is trained and evaluated k times.
# In each iteration, one of the k subsets is used as the test set, and the remaining k-1 subsets are used for training.
# This process is repeated until each subset has been used as a test set.
# The average performance across all iterations provides a more reliable estimate of the model's performance,
# helping to detect overfitting or underfitting.
#
# Hinglish:
#
# k-Fold Cross-Validation ek technique hai jo ek machine learning model ki performance ko assess karne mein istemal hoti hai.
# Dataset ko k subsets mein divide kiya jata hai, aur model ko k baar train aur evaluate kiya jata hai.
# Har iteration mein, ek kth subset ko test set ke roop mein istemal kiya jata hai, aur baki ke k-1 subsets ko
# training ke liye istemal kiya jata hai.
# Ye process tab tak dohraaya jata hai jab tak har subset ko test set ke roop mein istemal nahi ho jaata.
# Saare iterations ke average performance se model ki performance ka adhik satik estimate milta hai,
# jo overfitting ya underfitting ko detect karne mein madad karta hai.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
digits =load_digits()
data = pd.DataFrame(digits.data)
data['target'] = digits.target
X_train, X_test, Y_train, Y_test = train_test_split(data.drop(['target'], axis='columns'), data.target, test_size=0.2)

kf=KFold(n_splits=3)

for train_index ,test_index in kf.split([1,2,3,4,5,6,7,8,9]):

     print(train_index ,test_index )

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

folds = StratifiedKFold(n_splits=3)
scores_l = []  # Logistic regression
scores_svm = []
scores_rf = []  # Random Forest
#
# for train_index, test_index in kf.split(digits.data):
#     X_train, X_test = digits.data[train_index], digits.data[test_index]
#     y_train, y_test = digits.target[train_index], digits.target[test_index]
#
#     scores_l.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
#
#     # SVM
#     scores_svm.append(get_score(SVC(), X_train, X_test, y_train, y_test))
#
#     # Random Forest
#     scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))
#

scores_l =cross_val_score(LogisticRegression(),digits.data,digits.target)
scores_svm=cross_val_score(SVC(),digits.data,digits.target)
scores_rf=cross_val_score(RandomForestClassifier(n_estimators=50),digits.data,digits.target)
print("Logistic Regression Scores:", scores_l)
print("SVM Scores:", scores_svm)
print("Random Forest Scores:", scores_rf)
