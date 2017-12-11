import numpy as np
import random

def adaline_train(X,Y,eta=0.09,T=5000):
    #I suppose that there a column of 1's in X to calculate w0
    t=0
    m,n = X.shape
    # print "m"
    # print m
    w = np.zeros(n)
    while (t<T):
        r = random.randint(0,(m-1))
        h_w = np.dot( X[r] , w )
        # print("h_w: ",h_w)
        dim= Y[r] - h_w
        for i in range(0,n):
            w[i] += eta * (dim )*X[r][i]
        t+=1
    # print w
    return w

def adaline_test(X,Y,W):
    #Return accuracy
    m, n = X.shape
    good_prediction = 0
    for i in range(0,m):
        predict = np.dot(X[i],W)
        if predict * Y[i] > 0:
            good_prediction += 1
    return (float(good_prediction)/m)
