import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import random
import datetime


def convert_iso(date):
    date = date.split(sep='-')
    orddate = list()
    for item in date:
        item = item.lstrip('0')
        orddate.append(int(item))
    return datetime.date(orddate[0], orddate[1], orddate[2]).toordinal()


def algorithmize():
    # logistic regression, linear discriminant analysis, k-nearest neighbor
    # classification and regression tree, Gaussian Naive Bayes, support 
    # vector machine
    models = [
            ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
            ('LDA', LinearDiscriminantAnalysis()),
            ('KNN', KNeighborsClassifier()),
            ('CART', DecisionTreeClassifier()),
            ('NB', GaussianNB()),
            ('SVM', SVC(gamma='auto'))
            ]
    results = []
    names= []
    # Evaluate using each algorithm
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        
    return (results, names)


def main():
    data = pd.read_csv('data/AAPL.csv', usecols=['Date', 'Adj Close'])
    arr = data.values
    x = arr[:, 0]
    y = arr[:, 1]
    validation_size = 0.2
    seed = random.randint(1, 2**32-1)
    x_train, x_validation, y_train, y_validation = train_test_split(
            x, y, test_size=validation_size, random_state=seed)
    print(convert_iso('2013-05-01'))
    
if __name__ == '__main__':
    main()