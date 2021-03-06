import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, add_dummy_feature
try:
    from sklearn.model_selection import train_test_split    # sklearn > ...
except:
    from sklearn.cross_validation import train_test_split


def preprocess_ionosphere(data_folder,normalize=True):
    target_column = 34
    def discretize_func(s):
        if s == 'g':
            return 1
        else:
            return -1
    data_file = data_folder + "ionosphere.data"
    data = pd.read_csv(data_file, header = None) #Reading
    target = pd.DataFrame(data[target_column]) #Y
    features = data.drop([target_column], axis= 1) #X
    #tranform categorical to discrete
    y = target.applymap(discretize_func)
    #Add of a column of 1's to X
    scaler = StandardScaler()
    normFeatures = add_dummy_feature(scaler.fit_transform(features))
    x = pd.DataFrame(normFeatures )

    return x,y

def preprocess_fungus(data_folder):
    target_column = 0
    def discretize_func(s):
        if s == 'e':
            return 1
        else:
            return -1
    data_file = data_folder + "agaricus-lepiota.data"
    data = pd.read_csv(data_file, header = None) #Reading
    target = pd.DataFrame(data[target_column]) #Y
    features = data.drop([target_column], axis= 1) #X
    #tranform categorical to discrete target
    y = target.applymap(discretize_func)
    #tranform categorical to discrete features
    lenc = LabelEncoder()
    num_features = features.apply(lenc.fit_transform)
    print num_features.values
    scaler = StandardScaler()
    normFeatures = add_dummy_feature(scaler.fit_transform(num_features))
    x = pd.DataFrame(normFeatures )
    print x.values
    return x,y

def preprocess_wisconsin(data_folder):
    target_column = 10
    def discretize_func(s):
        if s == 2:
            return 1
        else:
            return -1
    data_file = data_folder + "breast-cancer-wisconsin.data"
    data = pd.read_csv(data_file, header = None) #Reading
    data = data.drop([0], axis=1)
    # print data.values
    target = pd.DataFrame(data[target_column]) #Y
    features = data.drop([target_column], axis= 1) #X
    #tranform categorical to discrete target
    y = target.applymap(discretize_func)
    #tranform categorical to discrete features
    lenc = LabelEncoder()
    num_features = features.apply(lenc.fit_transform)
    # print num_features.values
    scaler = StandardScaler()
    normFeatures = add_dummy_feature(scaler.fit_transform(num_features))
    x = pd.DataFrame(normFeatures )
    # print x.values
    return x,y

def preprocess_wdbc(data_folder):
    target_column = 1
    def discretize_func(s):
        if s == 'M':
            return 1
        else:
            return -1
    data_file = data_folder + "wdbc.data"
    data = pd.read_csv(data_file, header = None) #Reading
    data = data.drop([0], axis=1)
    # print data.values
    target = pd.DataFrame(data[target_column]) #Y
    features = data.drop([target_column], axis= 1) #X
    #tranform categorical to discrete target
    y = target.applymap(discretize_func)
    # print target.values
    #tranform categorical to discrete features
    lenc = LabelEncoder()
    num_features = features.apply(lenc.fit_transform)
    # print num_features.values
    scaler = StandardScaler()
    normFeatures = add_dummy_feature(scaler.fit_transform(num_features))
    x = pd.DataFrame(normFeatures )
    # print x.values
    return x,y

def preprocess_wpbc(data_folder):
    target_column = 1
    def discretize_func(s):
        if s == 'R':
            return 1
        else:
            return -1
    data_file = data_folder + "wpbc.data"
    data = pd.read_csv(data_file, header = None) #Reading
    data = data.drop([0], axis=1)
    # print data.values
    target = pd.DataFrame(data[target_column]) #Y
    features = data.drop([target_column], axis= 1) #X
    #tranform categorical to discrete target
    y = target.applymap(discretize_func)
    # print target.values
    #tranform categorical to discrete features
    lenc = LabelEncoder()
    num_features = features.apply(lenc.fit_transform)
    # print num_features.values
    scaler = StandardScaler()
    normFeatures = add_dummy_feature(scaler.fit_transform(num_features))
    x = pd.DataFrame(normFeatures )
    # print x.values
    return x,y

def split_train_test(x, y, testSize=0.4):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testSize)

    XTrain = X_train.values
    XTest = X_test.values
    YTrain = y_train.values
    YTest = y_test.values

    return XTrain, XTest, YTrain, YTest
