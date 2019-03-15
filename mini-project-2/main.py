import pandas as pd
import sklearn.model_selection


def main():
    data = pd.read_csv('iris.csv', names=['sepal length', 'sepal width',
                                          'petal length', 'petal width',
                                          'class'])
    
    pd.plotting.scatter_matrix(data)
    
    arr = data.values
    x = arr[: ,0:4]
    y = arr[: ,4]
    validation_size = 0.2
    seed = 7
    x_train, x_validation, y_train, y_validation = sklearn.model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed)

    # TODO: logistic regression, linear discriminant analysis, k-nearest neighbor
    # classification and regression tree, Gaussian Naive Bayes, support vector machine

if __name__ == '__main__':
    main()
