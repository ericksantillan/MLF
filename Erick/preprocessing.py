import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, add_dummy_feature
try:
    from sklearn.model_selection import train_test_split    # sklearn > ...
except:
    from sklearn.cross_validation import train_test_split

def preprocess_ionosphere(data_folder,target_column, normalize=True):
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
    features_p = add_dummy_feature(features)
    x = pd.DataFrame(features_p )

    return x,y

def split_train_test(x, y, testSize=0.4):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testSize)

    XTrain = X_train.values
    XTest = X_test.values
    YTrain = y_train.values
    YTest = y_test.values

    return XTrain, XTest, YTrain, YTest
