import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('mall_customers.csv')
    df = pd.DataFrame([[data['Age'].tolist()], [data['Spending Score'].tolist()]]).transpose()
    scatter = data.plot.scatter(x="Age", y="Spending Score")
    
if __name__ == '__main__':
    main()