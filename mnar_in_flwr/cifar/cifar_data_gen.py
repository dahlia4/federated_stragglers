from scipy.special import expit
import os
from vae_gen import VAE
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


predict_model = tf.keras.models.load_model("cat_ship_model.keras")
def CNN_decide(image):
    p = predict_model(image)[0][0]
    if p > 0.5:
        return 1
    else:
        return 0

def load_mnist_data():
    """
    Function to load MNIST data.
    Description of the data is here https://en.wikipedia.org/wiki/MNIST_database
    """

    (Xtrain, Ytrain), (Xtest, Ytest) = keras.datasets.mnist.load_data()

    # Normalize (divide by 255) input data
    Xtrain = Xtrain.astype("float32") / 255
    Xtest = Xtest.astype("float32") / 255

    return Xtrain, Ytrain, Xtest, Ytest

def load_cifar_data():
    (Xtrain, Ytrain), (Xtest, Ytest) = keras.datasets.cifar10.load_data()

    return Xtrain, Ytrain, Xtest, Ytest
def filter_classes_old(Xmat, Y, class0=3, class1=5):
    """
    Function to filter the data down to two classes
    """

    idxs = (Y == class0) | (Y == class1)
    Y_filtered = Y[idxs]
    return Xmat[idxs], np.array([0 if y == class1 else 1 for y in Y_filtered])

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

def generate_vae_point(D1,D2):
    X = D1 + np.random.normal(0, 2, 1)
    Y = np.random.normal(D2 - 2 * D1, 1)
    Z = 2 * D2 - np.random.uniform(0, 4, 1)
    A = np.random.uniform(0,2,1) - D1

    O1 = (expit(2 * X * Z + 2 * Z) * 2) - 1.0
    O2 = (expit(4 * Y * A + A * 2 + Y) * 2) - 1.0
    return O1, O2

def generate_point(D1, D2):
    #Fit a variational autoencoder on the mnist data, requires noise vectors, use this to generate digits
    #keras vae
    X = D1 + np.random.normal(0, 2, 1)
    Y = np.random.normal(D2 - 2 * D1, 1)
    Z = 2 * D2 - np.random.uniform(0, 2, 1)
    
    O1 = np.random.binomial(1, expit(2 * X * Y + 2 * Y + 2 * Z), 1)
    return O1

def piecewise_decide_class(x,y):
    #return true if 5, false if 3                                               
    if (x < -.75 and y > 0.3) or (-0.75 <= x < -0.25 and y > 0.25) or (-0.25 <= x < -0.05 and y > 0.15) or (-0.05 <= x < 0.25 and y > 0.05) or (0.25 <= x < 0.55 and y > 0.0) or (0.55 <= x < 0.8 and y > -0.05) or (0.8 <= x < 0.9 and y > -0.15) or (x >= 0.9 and y > -0.25):
        return True

def linear_decide_class(point):
    return clf.predict([point])[0]



vae_model = VAE(latent_space=128)
vae_model.build((None, 32, 32, 3))
vae_model.load_weights("VAE_CIFAR10_last_epoch_32.h5")
def get_cifar_vae_point(in_list):
    tensor_list = tf.convert_to_tensor([in_list], dtype=tf.float32)
    point = vae_model.decoder(tensor_list)
    Y = CNN_decide(point) 
    #Return the point and the output
    return point[0].numpy(),Y
    

def generate_demographics():
    D1 = np.random.binomial(1, 0.5, 1)[0]
    D2 = np.random.binomial(1, 0.5, 1)[0]
    return (D1, D2)

def dict_to_string(train_dict,out_name):
    train_dict_string = "{"
    other_start = True
    for client in train_dict:
        if not other_start:
            train_dict_string += ","
        else:
            other_start = False
        train_dict_string += f"{str(client)}: "
        train_dict_string += "["
        start = True
        for array in train_dict[client]:
            if not start:
                train_dict_string += ","
            else:
                start = False
            train_dict_string += f"({array[1]}, np.array({str(array[0].tolist())}))"
        train_dict_string += "]"
    train_dict_string += "}"
    return "import numpy as np \n" + out_name +  " = " + train_dict_string


ANCHOR_VECTORS = {
    (0, 0): np.random.RandomState(42).randn(128),
    (0, 1): np.random.RandomState(43).randn(128),
    (1, 0): np.random.RandomState(44).randn(128),
    (1, 1): np.random.RandomState(45).randn(128),
}
W = np.random.RandomState(99).randn(4, 128)

def generate_vae_point(D1, D2, latent_dim=128, noise_scale=0.8):

    X = D1 + np.random.normal(0, 1.5)
    Y = np.random.normal(0.5 * D2 - 0.5 * D1, 1)
    Z = D2 - np.random.uniform(0, 2)
    A = np.random.uniform(0, 2) - D1

    intermediary = np.array([
        expit(X * Z + Z) * 2 - 1.0,
        expit(2 * Y * A + A + Y) * 2 - 1.0,
        expit(X * A - Y * Z) * 2 - 1.0,
        expit(Y * Z + D1 - 0.5 * D2) * 2 - 1.0
    ])

    latent_shift = intermediary @ W
    anchor = ANCHOR_VECTORS[(D1, D2)]
    noise = np.random.normal(0, noise_scale, latent_dim)

    latent_point = anchor * 0.5 + latent_shift * 0.5 + noise * 1.0
    
    return latent_point

def get_vae_dict(num_clients,demographic_dict,training_num):
    train_dict = dict()
    ones = 0
    zeroes = 0
    #O1_avg = []
    #O2_avg = []
    for i in range(training_num):
        print(i)
        for j in range(num_clients):
            D1,D2 = demographic_dict[j]
            latent_point = generate_vae_point(D1,D2)
            #O1_avg.append(O1)
            #O2_avg.append(O2)
            #need to update this function
            point,Y = get_cifar_vae_point(latent_point)
            if Y == 0:
                zeroes += 1
            elif Y == 1:
                ones += 1
            if j not in train_dict:
                train_dict[j] = [(point,Y)]
            else:
                a =(point,Y)
                train_dict[j].append(a)
    return train_dict, zeroes, ones#, (sum(O1_avg)/len(O1_avg)), (sum(O2_avg)/len(O2_avg))

def get_dict(Xtrain_flat0,Xtrain_flat1,num_clients,demographic_dict):
    training_num = ((min(len(Xtrain_flat0),len(Xtrain_flat1))*2))//num_clients
    #training_num = 1                                                                                                                                                                                         
    train_dict = dict()
    for i in range(training_num):
        for j in range(num_clients):
            D1, D2 = demographic_dict[j]
            #print("generating point")
            O = generate_point(D1,D2)
            #print("generated point")
            if len(Xtrain_flat0) > 0 and len(Xtrain_flat1) > 0:
                #print("getting point")
                Xtrain_flat0,Xtrain_flat1,point,Y = get_point(Xtrain_flat0,Xtrain_flat1,O)
                #print("got point")
                if j not in train_dict:
                    train_dict[j] = [(point,Y)]
                else:
                    a = (point,Y)
                    train_dict[j].append(a)
    return train_dict

def get_clients(num_clients):
    demographic_dict = dict()
    for i in range(num_clients):
        demographic_dict[i] = generate_demographics()

    #train_dict = get_dict(Xtrain_flat0,Xtrain_flat1,num_clients,demographic_dict)
    #test_dict = get_dict(Xtest_flat0,Xtest_flat1,num_clients,demographic_dict)
    train_dict, train_zeroes, train_ones, O1_avg_train, O2_avg_train = get_cifar_vae_dict(num_clients,demographic_dict,training_num)
    test_dict, test_zeroes, test_ones, O1_avg_test, O2_avg_test = get_cifar_vae_dict(num_clients,demographic_dict,testing_num)

    
    train_dict_string = dict_to_string(train_dict,"in_data")
    test_dict_string = dict_to_string(test_dict,"in_test")
    with open("myclient/mnist_test.py","w") as writefile:
        writefile.write(test_dict_string)
    with open("myclient/mnist_train.py","w") as writefile:
        writefile.write(train_dict_string)
    with open("myclient/demographics.py","w") as writefile:
        writefile.write("demographic_dict = " + str(demographic_dict))
    demographic_dict_string = "demographic_dict = " + str(demographic_dict)
    return train_dict_string, demographic_dict_string, test_dict_string

def get_vae_clients(num_clients,training_num,testing_num):
    demographic_dict = dict()
    for i in range(num_clients):
        demographic_dict[i] = generate_demographics()

    train_dict, train_zeroes, train_ones = get_vae_dict(num_clients,demographic_dict,training_num)
    test_dict, test_zeroes, test_ones = get_vae_dict(num_clients,demographic_dict,testing_num)

    #with open("zeroes_and_ones.txt","a") as writefile:
    #    writefile.write(f"Zeroes: {str(train_zeroes)}, Ones: {str(train_ones)}, test zeroes: {test_zeroes} test_ones: {test_ones}, O1_avg_train {O1_avg_train}, O2_avg_train {O2_avg_train}, O1_avg_test {O1_avg_test} O2_avg_test {O2_avg_test}\n")
    train_dict_string = dict_to_string(train_dict,"in_data")
    test_dict_string = dict_to_string(test_dict,"in_test")
    
    with open("myclient/cifar_test.py","w") as writefile:
        writefile.write(test_dict_string)
    with open("myclient/cifar_train.py","w") as writefile:
        writefile.write(train_dict_string)
    with open("myclient/demographics.py","w") as writefile:
        writefile.write("demographic_dict = " + str(demographic_dict))
    demographic_dict_string = "demographic_dict = " + str(demographic_dict)
    return train_dict_string, demographic_dict_string, test_dict_string


train_dict_string = ""
demographic_dict_string = ""
test_dict_string = ""
def return_train_dict_string():
    return train_dict_string

def return_demographic_dict_string():
    return demographic_dict_string

def return_test_dict_string():
    return test_dict_string
if __name__ == "__main__":
    
    #train_dict_string, demographic_dict_string, test_dict_string = get_clients(int(sys.argv[1]))
    training_num = 100
    testing_num = 50
    train_dict_string, demographic_dict_string, test_dict_string = get_vae_clients(int(sys.argv[1]),training_num,testing_num)
    print("succeeded!")
#    print("starting server")
#    server = SimpleXMLRPCServer(("sysnet24.cs.williams.edu", 8000))
#    server.register_function(return_train_dict_string, "get_string")
#    server.register_function(return_demographic_dict_string, "get_demographics")
#    server.register_function(return_test_dict_string, "get_test")
#    print("serving")
#    server.serve_forever()
