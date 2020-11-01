# Malware Detection on IoT devices using Grey Wolf Optimization-MKSVM
In this project, we use Grey Wolf Optimization for feature selection and multi-kernel SVM for classification of a dataset. Two kernels of BFR and Linear are used, and you can change the kernels if you want (in MultiKernelSVM.py). The GWO creates the results in the numbers between 0 and 1, and we choose the features based on these numbers. We receive the numbers in method classify in MultiKernelSVM.py and create a model based on selected features. Afterward, we test this model and pass the accuracy as the fitness number to the GWO.

# How to use?
First, you need to change the param method and set the numbers of your features in your dataset (in optimizer.py). You can change the path of your dataset in the MultiKernelSVM.py.  In the next step, just run the optimizer file.

# Refrence
You can cite our paper:
https://ieeexplore.ieee.org/document/9205853

Note: This code is not the final version.
