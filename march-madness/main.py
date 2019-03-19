import pandas as pd
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
    x1 = data.loc[0, 'WFGM':'WPF']
    x2 = data.loc[0, 'LFGM':'LPF']
    # TODO: x = x1 - x2
    # Create third array of differences
#    y = data.loc[0, 'WTeamID']
#    x_train, x_validation, y_train, y_validation = train_test_split(
#            x, y, random_state=seed)
#    
#    models = [
#            ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
#            ('LDA', LinearDiscriminantAnalysis()),
#            ('KNN', KNeighborsClassifier()),
#            ('CART', DecisionTreeClassifier()),
#            ('NB', GaussianNB()),
#            ('SVM', SVC(gamma='auto'))
#            ]
#    results = []
#    names= []
#    
#    for name, model in models:
#        kfold = KFold(n_splits=10, random_state=seed)
#        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
#        results.append(cv_results)
#        names.append(name)


if __name__ == '__main__':
    main()