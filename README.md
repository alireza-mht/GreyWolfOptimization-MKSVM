# Grey Wolf Optimization-MKSVM
Using Grey Wolf Optimization for feature selection and multi-kernel SVM for classification of data-set.
In this project, we use Grey Wolf Optimization for feature selection of a dataset. After that, we use a multi-kernel SVM to create the model. Two kernels of BFR and Linear are used in this project and you can change the kernels if you want. The GWO creates results in the numbers between 0 and 1 and we choose the features based on these numbers. We create a model based on this feature selection. Afterward, test this model and pass the accuracy as the fitness number to the GWO.

# How to use?
First, you need to change the param method and set the numbers of your features in your dataset. in the next step, just run the optimizer file.
