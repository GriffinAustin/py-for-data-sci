import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    print('loading data...')
    data = pd.read_csv('ks-projects-201801.csv') 
    print('data successfully loaded')
    bar_graph = data.plot.bar(x='category', y='pledged', rot=0)
    plt.show()
    

if __name__ == '__main__':
    main()
