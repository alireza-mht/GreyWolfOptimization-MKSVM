# Import modules
import numpy as np
import seaborn as sns
import pandas as pd
#import pyswarms
import pyswarms as ps
from sklearn.datasets import make_classification

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

df = pd.read_csv('merged_dataset.csv', delimiter =',' , encoding='utf-8')
print(df.head())
print(df.dtypes)
print(df.keys())
# Plot toy dataset per feature
k = df['BWRate']
X, y = make_classification(n_samples=100, n_features=15, n_classes=3,
                           n_informative=4, n_redundant=1, n_repeated=2,
                           random_state=1)

df = pd.DataFrame(X)
df['labels'] = pd.Series(y)
sns.pairplot(df, hue='labels');