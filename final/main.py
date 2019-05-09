import pandas as pd
import matplotlib.pyplot as plt


# TODO: Remove '+' from 'Number of Installs' to make it numeric


def main():
    data = pd.read_csv('googleplaystore.csv')
    categories = data['Category'].value_counts()
    labels = categories.index
    values = categories.values
    
    patches, texts = plt.pie(values, labels=labels)
    plt.legend(patches, labels, loc='best')
    plt.axis('equal')
    
    plt.show()

if __name__ == '__main__':
    main()