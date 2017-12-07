# Perceptron algorithm
# m number of exemples
# n number of features
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import preprocess_ionosphere, split_train_test
from perceptron2 import perceptron_train, perceptron_test


print("Preprocessing....")
x, y= preprocess_ionosphere("../Databases/",34)
print("...Finished")

repeat = 20
eta_values = [0.0001,0.001,0.01,0.1,1,10,100,1000]
mean_accu = []
var_accu = []
for et in eta_values:
    sum_accu = []

    for k in range(0,repeat):
        XTrain, XTest, YTrain, YTest = split_train_test(x,y)

        W = perceptron_train(XTrain,YTrain, eta = et)
        # print W

        accu = perceptron_test(XTest,YTest,W)
        # print("Accuracy of "+str(k)+"  test: " + str(accu))
        sum_accu += [accu]

    print ("Eta = "+str(et)+", average accuracy: "+ str(np.mean(sum_accu)) + ", variance: "+str(np.var(sum_accu)))
    mean_accu +=[np.mean(sum_accu)]
    var_accu += [np.var(sum_accu)]
    print mean_accu

#pyplot
print mean_accu
plt.plot(eta_values, mean_accu)
plt.show()
