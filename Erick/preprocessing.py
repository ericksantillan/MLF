import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, add_dummy_feature
try:
    from sklearn.model_selection import train_test_split    # sklearn > ...
except:
    from sklearn.cross_validation import train_test_split

def preprocess(data_file,target_column, discretize_func, testSize=0.4, normalize=True):
    data = pd.read_csv(data_file, header = None) #Reading
    target = pd.DataFrame(data[target_column]) #Y
    features = data.drop([target_column], axis= 1) #X
    #tranform categorical to discrete
    y = target.applymap(discretize_func)
    #Add of a column of 1's to X
    features_p = add_dummy_feature(features)
    x = pd.DataFrame(features_p )
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testSize)
    #Converting to np.array
    XTrain = X_train.values
    XTest = X_test.values
    YTrain = y_train.values
    YTest = y_test.values
    return XTrain, XTest, YTrain, YTest
