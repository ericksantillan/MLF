# Perceptron algorithm
# m number of exemples
# n number of features
import numpy as np
import random
import pandas as pd

from preprocessing import preprocess_ionosphere, split_train_test
from perceptron2 import perceptron_train, perceptron_test


print("Preprocessing....")
x, y= preprocess_ionosphere("../Databases/",34)
print("...Finished")

sum_accu = 0.0
repeat = 100
for k in range(0,repeat):
    XTrain, XTest, YTrain, YTest = split_train_test(x,y)

    W = perceptron_train(XTrain,YTrain)
    # print W

    accu = perceptron_test(XTest,YTest,W)
    print("Accuracy of "+str(k)+"  test: " + str(accu))
    sum_accu += accu

print ("Average accuracy: "+ str(sum_accu))


#Scale data
