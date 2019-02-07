import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class GraphType():
    HIST = 0
    LINE = 1


class ManipData(object):
   
    
    def __init__(self, df):
        self.df = df
    
    def clean(self):
        self.df = self.df.dropna()
        
    def graph(self, g, **kwargs):
        if g == GraphType.HIST:
            self.hist = self.df.hist(**kwargs)
        if g == GraphType.LINE:
            self.line = self.df.plot.line(**kwargs)


def main():
    # Example of usage
    d = pd.DataFrame({
       'pig': [20, 18, 489, 675, 1776],
       'horse': [4, 25, 281, 600, 1900]
      }, index=[1990, 1997, 2003, 2009, 2014])
    x = ManipData(pd.DataFrame(data=d))
    x.clean()
    x.graph(GraphType.LINE, subplots=True)
    plt.show()


if __name__ == '__main__':
    main()