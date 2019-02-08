# Import modules
import sys
import pandas as pd
from komd import *
from sklearn import svm
import data_preprocessing as pre
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
class MultiKernelSVM():

    def __init__(self):
        self.df = pd.read_csv('merged_dataset.csv', delimiter=',',
                         encoding='utf-8')
        self.df = self.df.sample(20000)
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
        self.Y = self.df['SystemStatus(SystemClass)']
        self.X = self.df.drop(['SystemStatus(SystemClass)'],axis=1)
        #.to_numpy(dtype='int32')
        #.to_numpy()
        #self.X = pre.rescale(self.X)
        #self.X = pre.normalization(self.X)
        # train/test split
        from sklearn.model_selection import train_test_split

        X, Y = validation.check_X_y(self.X, self.Y, dtype=np.float64, order='C', accept_sparse='csr')
        # ohe = OneHotEncoder(sparse=False)
        #
        # self.X = ohe.fit_transform(self.X)
        # print(ohe.get_feature_names())
        # self.Y = ohe.fit_transform()


    def fit(self , l):
        print("start to create the fit")
        return classify(self.X,self.Y , l)


def classify(X , Y , l ):
    X_temp = X
    keys = X_temp.keys()
    p = keys[0]
    i =0
    for temp in l:
        if (temp<0):
            X_temp = X_temp.drop(keys[i],axis=1)
            p = X_temp.keys()
        i+=1
    X_temp = X_temp.to_numpy()
    Y = Y.to_numpy(dtype='int32')
    X_temp = pre.rescale(X_temp)
    X_temp = pre.normalization(X_temp)
    #kernel=my_kernel(X,Y), C=0.1 , cache_size=200 , decision_function_shape='ovr'
    Xtr, Xte, Ytr, Yte = train_test_split(X_temp, Y, test_size=.5, random_state=42)
    clf = svm.SVC(kernel=my_kernel,gamma='scale', decision_function_shape='ovo')
    global R
    R = Ytr

    clf.fit(Xtr,Ytr)
    R = Yte
    s = clf.predict(Xte)
    accuracy = accuracy_score(Yte, s)
    print(accuracy)
    return accuracy;

def my_kernel(X , K):
    global R
    bfr = KOMD()
    bfrKernel = bfr.returnKernel(X, R)

    linear = KOMD(kernel="linear")
    LinearKernel = linear.returnKernel(X, R)

    avaerage = 0.5 * (LinearKernel + bfrKernel)
    return avaerage



