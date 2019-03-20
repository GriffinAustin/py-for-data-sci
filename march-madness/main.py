import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random


def mean(data):
    return sum(data) / len(data)
    


def main():
    data = pd.read_csv('datafiles/NCAATourneyDetailedResults.csv')
    seed = random.randint(1, 2**32-1)
    
    y_vals = list() # Either 0 or 1, losing and winning respectively
    x_vals = list()
    for index, row in data.iterrows():
        winner = row['WFGM':'WPF']
        loser = row['LFGM':'LPF']
        team1 = random.randint(0, 1) # Randomly assign team 1
        if team1: # if team1 is the loser
            difference = loser.subtract(winner.values)
            y_vals.append([0])
        else: # if team1 is the winner
            difference = loser.subtract(winner.values)
            y_vals.append([1])
        x_vals.append(difference)
        
    y = pd.DataFrame(y_vals)
    x = pd.DataFrame(x_vals)   
    
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=seed)
    
    # Analyze algorithms
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
    
    for name, model in models:
        kfold = KFold(n_splits=3, random_state=seed)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
    
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    
    # Choose algorithm
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    predictions = nb.predict(x_validation)
    print(accuracy_score(y_validation, predictions))
    print(confusion_matrix(y_validation, predictions))
    print(classification_report(y_validation, predictions))
    
    # Make predictions
    x1 = data.loc[0, 'WFGM':'WPF']
    x2 = data.loc[0, 'LFGM':'LPF']
    x_new = x1.subtract(x2.values)
    x_new = x_new.to_frame()
    y_new = nb.predict(x_new)
    print(y_new)
    print(mean(y_new))
    print(x)


if __name__ == '__main__':
    main()
