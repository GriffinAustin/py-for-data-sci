import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# TODO: Remove '+' from 'Number of Installs' to make it numeric

def breakdown(data):
    categories = data['Category'].value_counts()
    labels = categories.index
    values = categories.values
    
    patches, texts = plt.pie(values, labels=labels)
    plt.legend(patches, labels, loc='lower right')
    plt.axis('equal')
    plt.title('App Market Breakdown')
    plt.show()
    
    
def rating(data):
    ratings = data['Rating']
    plt.hist(ratings, 41, range=(1, 5))
    plt.title('App Ratings')
    plt.show()
    print('Mean:', ratings.mean())
    print('Median:', ratings.median())
    print('Standard Deviation', ratings.std())
    
    
def scatter(data):
    data['Rating'].hist(by=data['Category'], range=(1, 5), bins=41)
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    
def main():
    data = pd.read_csv('googleplaystore.csv') 
    breakdown(data)
    rating(data)
    scatter(data)
    
    
if __name__ == '__main__':
    main()