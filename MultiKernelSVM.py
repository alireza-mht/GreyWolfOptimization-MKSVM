
"""
Created on Tue Jan 1 15:50:25 2019

@author: َAlireza
"""


# Import modules

import pandas as pd
from komd import *
from sklearn import svm
import data_preprocessing as pre
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class MultiKernelSVM():

    def __init__(self):
        self.df = pd.read_csv('merged_dataset.csv', delimiter=',',
                         encoding='utf-8')
        self.df = self.df.sample(20000)
        self.X = None
        self.Y = None


    def preprocessing(self):
        print("start preprocessing the dataset")
        cleanup_label = {"SystemStatus(SystemClass)": {"DDoS_severe_normal_maxload": 0,
                                                       "DDoS_mild_normal_maxload": 1,
                                                       "DDoS_degrading_normal_maxload": 2,
                                                       "normal_max_load": 3,
                                                       "normal_average_load": 4}}
        self.df.replace(cleanup_label,inplace = True)
        self.Y = self.df['SystemStatus(SystemClass)']
        self.X = self.df.drop(['SystemStatus(SystemClass)'],axis=1)

        X, Y = validation.check_X_y(self.X, self.Y, dtype=np.float64, order='C', accept_sparse='csr')


    def fit(self , l):
        print("start to create the fit")
        return classify(self.X,self.Y , l)


def classify(X , Y , l ):
    X_temp = X
    keys = X_temp.keys()
    p = keys[0]
    i =0

    #feature selection based on the GWO nummbers
    for temp in l:
        if (temp<0):
            X_temp = X_temp.drop(keys[i],axis=1)
            p = X_temp.keys()
        i+=1
    X_temp = X_temp.to_numpy()
    Y = Y.to_numpy(dtype='int32')

    #normalising data
    X_temp = pre.rescale(X_temp)
    X_temp = pre.normalization(X_temp)

    #spliting test and train
    Xtr, Xte, Ytr, Yte = train_test_split(X_temp, Y, test_size=.5, random_state=42)

    #defining the SVM classifier and the kernel method
    clf = svm.SVC(kernel=my_kernel,gamma='scale', decision_function_shape='ovo')

    #defining the global variable for using in the my_kernel method
    global R
    R = Ytr

    #creating the model
    clf.fit(Xtr,Ytr)
    R = Yte

    #predicting the test data
    s = clf.predict(Xte)
    accuracy = accuracy_score(Yte, s)
    print(accuracy)
    return accuracy;

def my_kernel(X , K):
    global R

    #Creating the BFR kernel
    bfr = KOMD()
    bfrKernel = bfr.returnKernel(X, R)

    #Creating the Linear kernel
    linear = KOMD(kernel="linear")
    LinearKernel = linear.returnKernel(X, R)

    #Averaging two kernel
    avaerage = 0.5 * (LinearKernel + bfrKernel)
    return avaerage



