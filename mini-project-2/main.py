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


def algorithmize(data):
    # Prepare training data
    arr = data.values
    x = arr[: ,0:4]
    y = arr[: ,4]
    validation_size = 0.2
    seed = random.randint(1, 2**32-1)
    x_train, x_validation, y_train, y_validation = train_test_split(
            x, y, test_size=validation_size, random_state=seed)
    print(type(y_train))

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
    data = pd.read_csv('iris.csv', names=['sepal length', 'sepal width',
                                          'petal length', 'petal width',
                                          'class'])
    
    pd.plotting.scatter_matrix(data)
    
    evaluation = algorithmize(data)
        
    # Compare algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(evaluation[0])
    ax.set_xticklabels(evaluation[1])
    plt.show()
    
    # TODO: Make predictions

if __name__ == '__main__':
    main()
