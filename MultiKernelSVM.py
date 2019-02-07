# Import modules
import sys
import pandas as pd
from komd import *
from sklearn import svm
import data_preprocessing as pre
from sklearn.preprocessing import OneHotEncoder

class MultiKernelSVM():

    def __init__(self):
        self.df = pd.read_csv('merged_dataset.csv', delimiter=',',
                         encoding='utf-8')
        self.df = self.df.sample(3000)
        self.X = None
        self.Y = None
       # self.Y = df['SystemStatus(SystemClass)']
        #self.X = df.drop(['SystemStatus(SystemClass)'],axis=1)

    def preprocessing(self):
        print("start preprocessing the dataset")
        cleanup_label = {"SystemStatus(SystemClass)": {"DDoS_severe_normal_maxload": 0,
                                                       "DDoS_mild_normal_maxload": 1,
                                                       "DDoS_degrading_normal_maxload": 2,
                                                       "normal_max_load": 3,
                                                       "normal_average_load": 4}}
        self.df.replace(cleanup_label,inplace = True)
        self.Y = self.df['SystemStatus(SystemClass)'].to_numpy(dtype='int32')
        self.X = self.df.drop(['SystemStatus(SystemClass)'],axis=1).to_numpy()
        #self.X = pre.rescale(self.X)
        #self.X = pre.normalization(self.X)
        X, Y = validation.check_X_y(self.X, self.Y, dtype=np.float64, order='C', accept_sparse='csr')
        # ohe = OneHotEncoder(sparse=False)
        #
        # self.X = ohe.fit_transform(self.X)
        # print(ohe.get_feature_names())
        # self.Y = ohe.fit_transform()


    def fit(self , l):
        print("start to create the fit")
        classify(self.X,self.Y , l)


def classify(X , Y , l):
    #kernel=my_kernel(X,Y), C=0.1 , cache_size=200 , decision_function_shape='ovr'
    clf = svm.SVC(kernel=my_kernel,gamma='scale', decision_function_shape='ovo')
    clf.fit(X,Y)
    s = clf.predict(X)
    print(s)
    return 0;

def my_kernel(X , K):
    bfr = KOMD()
    bfrKernel = bfr.returnKernel(X, Y)

    linear = KOMD(kernel="linear")
    LinearKernel = linear.returnKernel(X, Y)

    avaerage = 0.5 * (LinearKernel + bfrKernel)
    return avaerage



a = MultiKernelSVM()
a.preprocessing()
Y = a.Y
a.fit(0)
