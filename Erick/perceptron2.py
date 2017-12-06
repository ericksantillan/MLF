import numpy as np
import random

def perceptron_train(X, Y, eta=1, T=5000):
    #I suppose that there a column of 1's in X to calculate w0
    t=0
    m,n = X.shape
    w = np.zeros(n)
    while (t<T):
        r = random.randint(0,(m-1))
        if Y[r] *( np.dot( X[r] , w ) ) <= 0:
            for i in range(0,n):
                w[i] += eta * Y[r]*X[r][i]
        t+=1
    return w

def perceptron_test(X,Y,W):
    #Return accuracy
    m, n = X.shape
    good_prediction = 0
    for i in range(0,m):
        predict = np.dot(X[i],W)
        if predict * Y[i] > 0:
            good_prediction += 1
    return (float(good_prediction)/m)
