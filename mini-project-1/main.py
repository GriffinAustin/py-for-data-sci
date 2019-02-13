import pandas as pd
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

    final_data.sort(key=lambda x: x[1])
    x_data = list()
    y_data = list()
    [x_data.append(x[0]) for x in final_data]
    [y_data.append(y[1]) for y in final_data]

    plt.bar(x_data, y_data)

    plt.title('Mean HF Score vs. Region')
    plt.xlabel('Regiosn')
    plt.ylabel('Mean HF Score')
    plt.xticks(rotation='vertical')
    plt.ylim(0, 10)

    plt.show()

    # Saves graph
    #plt.savefig('meanhfscore_vs_region.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
