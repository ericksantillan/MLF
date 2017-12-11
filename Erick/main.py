# Perceptron algorithm
# m number of exemples
# n number of features
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import preprocess_ionosphere, split_train_test
from perceptron2 import perceptron_train, perceptron_test
from adaline import adaline_train, adaline_test
from logistic import gradient_descent


print("Preprocessing....")
x, y= preprocess_ionosphere("../Databases/")
print("...Finished")

repeat = 20
T = [50,500,5000]
ln_T= [1,2,3]
eta_values = [0.001,0.01,0.1,1,10,100,1000]
ln_eta = [-3,-2,-1,0,1,2,3]
mean_accu = []
var_accu = []
for t in T:
# for et in eta_values:
    sum_accu = []

    for k in range(0,repeat):
        XTrain, XTest, YTrain, YTest = split_train_test(x,y)

        W = adaline_train(XTrain,YTrain,eta=0.001, T = t)
        # W = adaline_train(XTrain,YTrain, eta = et)
        # W = perceptron_train(XTrain,YTrain, eta = et)
        # W = perceptron_train(XTrain,YTrain, T = t)

        accu = adaline_test(XTest,YTest,W)
        # accu = perceptron_test(XTest,YTest,W)
        sum_accu += [accu]

    print ("T = "+str(t)+", average accuracy: "+ str(np.mean(sum_accu)) + ", variance: "+str(np.var(sum_accu)))
    # print ("Eta = "+str(et)+", average accuracy: "+ str(np.mean(sum_accu)) + ", variance: "+str(np.var(sum_accu)))
    mean_accu +=[np.mean(sum_accu)]
    var_accu += [np.var(sum_accu)]
    print mean_accu

#pyplot
print mean_accu
plt.plot(ln_T, mean_accu)
plt.ylabel('Accuracy')
plt.xlabel('ln(T)')
plt.axis([1, 3, 0.50, 1])
plt.show()
