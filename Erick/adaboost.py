# Perceptron algorithm
# m number of exemples
# n number of features
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, add_dummy_feature
from preprocessing import preprocess

def adaboost_train(X,Y,T=20):
        #I suppose that there a column of 1's in X to calculate w0
        t=0
        m,n = X.shape
        D = np.zeros(m)
        for k in range(0,m):
            D[k]= 1.0/m
        w = np.zeros(n)
        alpha = np.zeros(T)
        Ft = []
        while (t<T):
            weak_classif = weak_perceptron(X,Y,D)
            Ft += [weak_classif]
            eps=0.0
            Zt = 0.0
            for i in range(0,m):
                if weak_classif(X[i]) != Y[i]:
                    eps+= D[i]
            alphat = 0.5 * np.log( (1-eps)/eps   )
            alpha[t] = alphat
            #Calcul of Zt
            for i in range(0,m):
                Zt += D[i] * np.exp( -alphat * float(Y[i])*float(weak_classif(X[i])))
            #calcul of new Distribution
            for i in range(0,m):
                D[i] = ( D[i]*np.exp( -alphat * float(Y[i])*float(weak_classif(X[i]))) ) / Zt

            t+=1
            print("Adaboost t:"+str(t))

        def F(x):
            s=0
            for k in range(0,T):
                s += alpha[k]*Ft[k](x)
            if s<0:
                return -1
            else:
                return 1

        return F

def adaboost_test(X,Y,F):
    m,n = X.shape
    accu = 0.0
    for i in range(0,m):
        if Y[i]*F(X[i]) > 0:
            accu += 1.0
    return accu/m

def weak_perceptron(X,Y,D,eta=1,T=5000):
    #I suppose that there a column of 1's in X to calculate w0
    t=0
    m,n = X.shape
    w = np.zeros(n)
    while (t<T):
        r = np.random.choice(m, p=D)
        if Y[r] *( np.dot( X[r] , w ) ) <= 0:
            for i in range(0,n):
                w[i] += eta * Y[r]*X[r][i]
        t+=1
    def classif(Xi):
        pred = np.dot(Xi, w)
        if pred < 0:
            return 1
        else:
            return -1
    return classif

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

def perceptron_test(X,Y,W):    #I suppose that there a column of 1's in X to calculate w0
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
    #Return accuracy
    m, n = X.shape
    good_prediction = 0
    for i in range(0,m):
        predict = np.dot(X[i],W)
        if predict * Y[i] > 0:
            good_prediction += 1
    return (float(good_prediction)/m)

def tranform_categorical(s):
    if s == 'g':
        return 1
    else:
        return -1

# XTrain, XTest, YTrain, YTest = preprocess("../Databases/ionosphere.data",34, tranform_categorical)
# Preprocessing
print("READING DataBase....")
data = pd.read_csv("../Databases/ionosphere.data", header = None) #Reading
target = pd.DataFrame(data[34]) #Y
features = data.drop([34], axis= 1) #X

print("Preprocessing Data")
#tranform categorical to discrete
y = target.applymap(tranform_categorical)
y.head()
#Add of a column of 1's to X
features_p = add_dummy_feature(features)
x = pd.DataFrame(features_p )

sum_accu = 0
repeat = 100
for k in range(0,repeat):
#Splitting between training and testing
    try:
        from sklearn.model_selection import train_test_split    # sklearn > ...
    except:
        from sklearn.cross_validation import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    XTrain = X_train.values
    XTest = X_test.values
    YTrain = y_train.values
    YTest = y_test.values

    print(k)
    F = adaboost_train(XTrain, YTrain)
    # print W

    accu = adaboost_test(XTest, YTest,F)
    print("accuracy" + str(accu))
    sum_accu += accu

print sum_accu
