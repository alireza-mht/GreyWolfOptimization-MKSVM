# Import modules
import sys
import pandas as pd
from komd import *
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel

# from shogun.Evaluation import *lassification

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

class MultiKernelSVM():


    def __init__(self):
        df = pd.read_csv('merged_dataset.csv', delimiter=',',
                         encoding='utf-8')
        print(df.head())
        print(df.dtypes)
        print(df.keys())

        self.Y

        self.X = None
        self.Y = None

    def Preprocessing(self):
        print("start preprocessing the dataset")


    def fit(self , l):
        print("start to create the fit")
        classify(self.X,self.Y , l)


def my_kernel(self, X, Y):
    bfr = KOMD()
    bfrKernel = bfr.returnKernel(X, Y)

    linear = KOMD(kernel='linear')
    LinearKernel = linear.returnKernel(X, Y)

    return 0.5 * (LinearKernel + bfrKernel)

def classify(X , Y , l):
    clf = svm.SVC(kernel=my_kernel)
    clf.fit(X,Y)

    return 0;

