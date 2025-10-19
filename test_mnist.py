from scipy.special import expit

from tensorflow import keras
import numpy as np
from random import randint
from xmlrpc.server import SimpleXMLRPCServer

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

def filter_classes_old(Xmat, Y, class0=0, class1=1):
    """
    Function to filter the data down to two classes
    """

    idxs = (Y == class0) | (Y == class1)
    Y_filtered = Y[idxs]
    return Xmat[idxs], np.array([0 if y == class1 else 1 for y in Y_filtered])

def filter_classes(Xmat, Y, class0=0, class1=1):
    """                                                                                                                                                                                
    Function to filter the data down to two classes                                                                                                                                    
    """

    idxs0 = (Y == class0)
    Y_filtered0 = Y[idxs0]
    idxs1 = (Y == class1)
    Y_filtered1 = Y[idxs1]
    return Xmat[idxs0], Xmat[idxs1], np.array([0 for y in Y_filtered0]), np.array([1 for y in Y_filtered1])

def generate_point(D1, D2):
    X = D1 + np.random.normal(0, 2, 1)
    Y = np.random.normal(D2 - 2 * D1, 1)
    Z = 2 * D2 - np.random.uniform(0, 2, 1)
    
    O1 = np.random.binomial(1, expit(2 * X * Y + 2 * Y + 2 * Z), 1)
    return O1

def get_point(X0,X1,O):
    if O == 0:
        i = randint(0,len(X0)-1)
        point = X0[i]
        new_X0 = np.delete(X0,[i],axis=0)
        return (new_X0,X1,point,0)
    else:
        i = randint(0,len(X1)-1)
        point =	X1[i]
        new_X1 = np.delete(X1,[i],axis=0)
        return (X0, new_X1,point,1)

def generate_demographics():
    D1 = np.random.binomial(1, 0.5, 1)
    D2 = np.random.binomial(1, 0.5, 1)
    return (D1, D2)
    
def get_clients(num_clients):
    demographic_dict = dict()
    for i in range(num_clients):
        demographic_dict[i] = generate_demographics()
        
    Xtrain, Ytrain, Xtest, Ytest = load_mnist_data()
    Xtrain0, Xtrain1, Ytrain0, Ytrain1 = filter_classes(Xtrain, Ytrain)
    Xtest0, Xtest1, Ytest0, Ytest1 = filter_classes(Xtest, Ytest)
    
    # flattens the 28x28 images into 784 dimension vectors
    Xtrain_flat0, Xtest_flat0 = Xtrain0.reshape((len(Xtrain0), 784)), Xtest0.reshape((len(Xtest0), 784))
    Xtrain_flat1, Xtest_flat1 = Xtrain1.reshape((len(Xtrain1), 784)), Xtest1.reshape((len(Xtest1), 784))
    
    training_num = ((min(len(Xtrain_flat0),len(Xtrain_flat1))*2))//num_clients
    #training_num = 1
    train_dict = dict()
    for i in range(training_num):
        for j in range(num_clients):
            D1, D2 = demographic_dict[j]
            O = generate_point(D1,D2)
            if len(Xtrain_flat0) > 0 and len(Xtrain_flat1) > 0:
                Xtrain_flat0,Xtrain_flat1,point,Y = get_point(Xtrain_flat0,Xtrain_flat1,O)
                if j not in train_dict:
                    train_dict[j] = [(point,Y)]
                else:
                    a = (point,Y)
                    train_dict[j].append(a)
    #print(type(train_dict[0]))
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
    with open("test.py","w") as writefile:
        writefile.write("import numpy as np \ntemp = " + train_dict_string)
    train_dict_string = "import numpy as np \ntemp = " + train_dict_string
    demographic_dict_string = "demographic_dict = " + str(demographic_dict)
    return train_dict_string, demographic_dict_string
train_dict_string = ""
def return_train_dict_string():
    return train_dict_string

def return_demographic_dict_string():
    return demographic_dict_string
if __name__ == "__main__":
    
    train_dict_string, demographic_dict_string = get_clients(100)
    
    print("starting server")
    server = SimpleXMLRPCServer(("sysnet24.cs.williams.edu", 8000))
    server.register_function(return_train_dict_string, "get_string")
    server.register_function(return_train_dict_string, "get_demographics")
    print("serving")
    server.serve_forever()
