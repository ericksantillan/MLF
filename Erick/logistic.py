import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, add_dummy_feature
from preprocessing import preprocess


def logistic_function(X,Y,w):
    m, n = X.shape
    l = 0.0
    for j in range(0,m):
        l += np.log(  1 +  np.exp( -1.0*float(Y[j])*(np.dot(X[j], w) ) )  )
    return l/m

def f_grad(X,Y,w):
    m, n = X.shape    
    g = np.zeros(n)
    for j in range(0,m):
        if Y[j] >0:
            g += -X[j]/( 1 + np.exp(np.dot( X[j] , w ) ) )
        else:
            g += X[j]/( 1 + np.exp(-np.dot( X[j] , w ) ) )
    return g/m

def gradient_descent(X,Y,D=None):
    m, n = X.shape

    if D is None:
        D = np.zeros(m)
        for k in range(0,m):
            D[k]= 1.0/m

    gamma = 0.99
    epsilon = 0.01
    w = np.random.rand(n)
    while(  np.linalg.norm(f_grad(X,Y,w)) > epsilon   ):
        w += -gamma*f_grad(X,Y,w)
        # print np.linalg.norm(f_grad(X,Y,w))
    return w

def predict(Xi,w):
    p =  1.0 / ( 1 + np.exp(- np.dot(Xi, w) )  )
    if p > 0.5:
        return 1
    else:
        return -1


def logistic_test(X,Y,w):
    m, n = X.shape
    accu = 0.0
    for j in range(0,m):
        pred = predict(X[j], w)
        if pred * Y[j] > 0:
            accu +=1
    return accu/m


def tranform_categorical(s):
    if s == 'g':
        return 1
    else:
        return -1

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
repeat = 10
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
    w = gradient_descent(XTrain, YTrain)
    print(w)
    acc = logistic_test(XTest, YTest, w)
    print acc



print sum_accu
