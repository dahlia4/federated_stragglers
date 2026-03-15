from scipy.special import expit
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from tensorflow import keras
import tensorflow as tf
ops = tf
from keras import layers
import matplotlib.pyplot as plt
import random

from tensorflow import keras
import numpy as np
from random import randint
from xmlrpc.server import SimpleXMLRPCServer
import sys
import pickle

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression

(Xtrain, Ytrain), (Xtest, Ytest) = keras.datasets.mnist.load_data()

def filter_classes(Xmat, Y, class0=3, class1=5):
    """                                                                                                                                                                                                                            
    Function to filter the data down to two classes                                                                                                                                                                                
    """
    newY = Y.squeeze()

    idxs0 = (newY == class0)
    Y_filtered0 = newY[idxs0]
    idxs1 = (newY == class1)
    Y_filtered1 = newY[idxs1]

    x_return0 = Xmat[idxs0]
    x_return1 = Xmat[idxs1]

    array_0 =  np.array([0 for y in Y_filtered0])
    array_1 = np.array([1 for y in Y_filtered1])

    return x_return0, x_return1, array_0, array_1

Xtrain0, Xtrain1, Ytrain0, Ytrain1 = filter_classes(Xtrain, Ytrain)
Xtest0, Xtest1, Ytest0, Ytest1 = filter_classes(Xtest, Ytest)
Xtrain_flat0, Xtest_flat0 = Xtrain0.reshape((len(Xtrain0), 784)), Xtest0.reshape((len(Xtest0), 784))
Xtrain_flat1, Xtest_flat1 = Xtrain1.reshape((len(Xtrain1), 784)), Xtest1.reshape((len(Xtest1), 784))  


full_Xtrain = np.concatenate((Xtrain_flat0,Xtrain_flat1))
full_Ytrain = np.concatenate((Ytrain0,Ytrain1))
full_Xtest = np.concatenate((Xtest_flat0,Xtest_flat1))
full_Ytest = np.concatenate((Ytest0,Ytest1))

print(full_Xtrain.shape)
clf = LogisticRegression(max_iter = 2).fit(full_Xtrain, full_Ytrain)
pickle.dump(clf, open("model_clf_pickle", 'wb'))
print(clf.score(full_Xtest,full_Ytest))


print(clf.predict([full_Xtest[0]]))
