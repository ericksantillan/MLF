import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import preprocess_fungus,preprocess_wisconsin,preprocess_wdbc,  split_train_test
from perceptron2 import perceptron_train, perceptron_test

preprocess = preprocess_wdbc

print("Preprocessing....")
x, y= preprocess("../Databases/")
print("...Finished")
