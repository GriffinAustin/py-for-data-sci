import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import enum


class GraphType(enum.Enum):
    HIST = 0


class ManipData(object):
   
    
    def __init__(self, df):
        self.df = df
    
    def clean(self):
        self.df = self.df.dropna()
        
    def graph(iself, g):
        if g == 0:
            self.hist = self.df.hist()

d = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
    "toy": [np.nan, 'Batmobile', 'Bullwhip'],
    "born": [pd.NaT, pd.Timestamp("1940-04-25"), pd.NaT]})
x = ManipData(pd.DataFrame(data=d))
x.clean()
x.graph(GraphType.HIST)
print(int(GraphType.HIST))
plt.show()
