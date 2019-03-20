import pandas as pd
import numpy as nd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import random


def main():
    data = pd.read_csv('datafiles/NCAATourneyDetailedResults.csv')
    seed = random.randint(1, 2**32-1)
    
    # TODO: Load actual data instead of test data
    
    x1 = data.loc[0, 'WFGM':'WPF']
    x2 = data.loc[0, 'LFGM':'LPF']
    x = x1.subtract(x2.values)
    x = x.values[:]
    
    y = pd.DataFrame({'win': [1]*16})
    y['win'][2] = 0
    y['win'][1] = 0
    y['win'][5] = 0
    y['win'][7] = 0
    y = y.values[:]
    y = y.ravel()
    
    x = pd.DataFrame(x)
    
    x = x.transpose()
    x3 = x
    x = x.append(x.subtract(1))
    x = x.append(x.subtract(1))
    x = x.append(x.subtract(1))
    x = x.append(x.subtract(1))
        
    
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=seed)
    
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


if __name__ == '__main__':
    main()
