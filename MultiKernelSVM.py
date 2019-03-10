



import pandas as pd
import numpy as np
import data_preprocessing as pre

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel


class MultiKernelSVM():

    def __init__(self):

        self.df = pd.read_csv('./dataset/merged_dataset_weighted.csv', delimiter=',',
                       encoding='utf-8')
        self.df = self.df.sample(10000)


    def preprocessing(self):
         print("start preprocessing the dataset")

         cleanup_label = {"SystemStatus(SystemClass)": {"zero": 0,
                                                       "one": 1,
                                                       "two": 2,
                                                       "three": 3,
                                                       "four": 4}}
         self.df.replace(cleanup_label,inplace = True)

         self.Y = self.df['SystemStatus(SystemClass)']
         self.X = self.df.drop(['SystemStatus(SystemClass)'], axis=1)

    def fit(self , l):
        print("start to create the fit")
        return classify(self.X, self.Y, l)


def classify(X , Y, l):
    X_temp = X
    keys = X.keys()

    i = 0
    # feature selection based on the GWO nummbers
    for temp in l:
        if (temp==0):
            X_temp = X.drop(keys[i],axis=1)
            X = X.drop(keys[i],axis=1)
        i+=1

    # Y =
    # (Y, classes=[0, 1, 2, 3, 4])
    X_temp = X_temp.to_numpy()
    Y = Y.to_numpy(dtype='int32')

    # Y = Y[:,0]
    # H = Y[:,1]

    #normalising data
    X_temp = pre.rescale_01(X_temp)
    X_temp = pre.normalization(X_temp)


    #spliting test and train
    Xtr, Xte, Ytr, Yte = train_test_split(X_temp, Y, test_size=0.3)

    #defining the SVM classifier and the kernel method
    clf = svm.SVC(kernel=my_kernel)

    #creating the model
    clf.fit(Xtr,Ytr)
    # .decision_function(Xte)



    # accuracy = accuracy.mean()
    #Preform 10-flod cross validation
    # n_samples = iris.data.shape[0]
    # scores = cross_val_score(clf, X, Y, cv=5)
    # accuracy = scores.mean()
    #
    # predictions = cross_val_predict(clf, X, Y, cv=10)

    #predicting the test data
    s = clf.predict(Xte)
    accuracy = accuracy_score(Yte, s)

    # evaluation
    # binarise_result = label_binarize(result, classes=class_list)
    # binarise_labels = label_binarize(test_labels, classes=class_list)
    # generate_eval_metrics(binarise_result, 'tfidf_ada', binarise_labels)
    print(accuracy)
    # print(recall)
    return accuracy

def my_kernel(X , K):
    # Creating the RBF kernel
    bfrKernel= rbf_kernel(X, K , 0.1)
    bfrKernel= bfrKernel.astype(np.double)

    # Creating the poly kernel
    polyKernel = polynomial_kernel(X, K, degree=2.0, gamma=0.1, coef0=0.0)
    polyKernel = polyKernel.astype(np.double)

    # Creating the Linear kernel
    linearKernel = linear_kernel(X,K)
    linearKernel = linearKernel.astype(np.double)


    return (linearKernel + bfrKernel + polyKernel )/3



