import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Question: Is there a correlation between country region and average hf score?
def main():
    print('loading data...')
    data = pd.read_csv('hfi_cc_2018.csv')
    print('data successfully loaded')
    data = data.groupby('region')
    final_data = list()
    
    for name, group in data:
        final_data.append((name, group['hf_score'].mean()))
        
    final_data.sort(key=lambda x:x[1])
    x_data = list()
    y_data = list()
    [x_data.append(x[0]) for x in final_data]
    [y_data.append(y[1]) for y in final_data]
    
    plt.bar(x_data, y_data)
    plt.xlabel("Region")
    plt.ylabel("Mean HF Score")
    plt.xticks(rotation='vertical')
    plt.show()

if __name__ == '__main__':
    main()
