"""MyClient: A Flower / PyTorch app."""

import flwr
import torch
from flwr.client import NumPyClient, Client, ClientApp
from flwr.common import Metrics, Context, ConfigRecord, RecordDict
from .knobs import DEVICE, MISSING
from .net import Net, train, test, set_parameters, get_parameters
from torch.utils.data import Dataset, DataLoader
#from dataset_loader import load_datasets                                                                         
import pandas as pd
import numpy as np
from scipy.special import expit


# Define Flower Client and client_fn
class IntermediateDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MyClient(NumPyClient):
    """                                                                                                           
    To define NumPyClient: init, fit, evaluate, get_parameters                                                    
    """

    def __init__(self,net,context):
        self.client_state = context.state
        self.net = net
        self.properties = {}


        self.demographics = {}
        if "demographics" in self.client_state.config_records:
            self.demographics["D1"] = self.client_state.config_records["demographics"]["D1"]
            self.demographics["D2"] = self.client_state.config_records["demographics"]["D2"]
            
        dataset = self._generate_set(1)
        self.demographics = {
            "D1": dataset.reset_index()["D1"][0],
            "D2": dataset.reset_index()["D2"][0],
        }

        self.survey_dataset = dataset[["D1", "D2", "R", "S"]]
        
        #self.response = self.survey_dataset.reset_index(drop=True)["R"][0]
        #self.satisfied = self.survey_dataset.reset_index(drop=True)["S"][0]

        self.dataset = dataset[["X", "Y", "Z", "O1"]]


        self._generate_large_train_set(500)

    def get_parameters(self,config):
        #Return parameters the local model has                                                                    
        return get_parameters(self.net)

    def fit(self,parameters,config):
        #Set the local model to have the global parameters                                                        
        set_parameters(self.net,parameters)


        df = self.dataset.sample(n=20, replace=True)
        x_train = df.drop(["O1"], axis=1).to_numpy()
        y_train = df["O1"].to_numpy()

        train_dataset = IntermediateDataset(x_train,y_train)
        trainloader = DataLoader(train_dataset,batch_size = 1)
        #Train model locally for one epoch                                                                        
        train(self.net,trainloader,1)

        #Return updated parameters, number of examples used for training, dict with "metrics"                     
        return get_parameters(self.net),len(trainloader),{}

    def evaluate(self,parameters,config):
        #Set the local model to have the global parameters                                                        
        set_parameters(self.net,parameters)
        df = self._generate_set(10).drop(["R", "S", "D1", "D2"], axis=1)

        x_val = df.drop(["O1"], axis=1).to_numpy()
        y_val = df["O1"].to_numpy()
        val_dataset = IntermediateDataset(x_val,y_val)
        valloader = DataLoader(val_dataset,batch_size = 1)
        #Run our testing function                                                                                 
        loss, accuracy = test(self.net,valloader)
        #print(loss,accuracy)
        #Return loss, num_examples, and metrics dictionary                                                        
        return loss, len(valloader), {"accuracy": float(accuracy)}

    def get_properties(self, ins=None,config=None):
        temp_dict = self._generate_set(1).iloc[0].to_dict()
        if temp_dict["R"] == 0 and MISSING:
            temp_dict["S"] = -1
        return temp_dict
    def _generate_large_train_set(self, num_rows):
        """                                                                                                      \
                                                                                                                  
        This method generates a larger dataset for training purposes. NOTE: calling this method                  \
                                                                                                                  
        mutates the dataset attribute. 
"""
        large_dataset = self._generate_set(num_rows)
        #large_dataset = large_dataset[
        #    (large_dataset["R"] == self.response)
        #    & (large_dataset["S"] == self.satisfied)
        #]

        assert len(large_dataset) > 0
        
        self.dataset = large_dataset[["X", "Y", "Z", "O1"]]

    def _generate_set(self, n_samples):
        if len(self.demographics) == 0:
            D1 = np.random.binomial(1, 0.5, 1)
            D2 = np.random.binomial(1, 0.5, 1)
            self.demographics = {"D1": D1, "D2": D2}
            self.client_state.config_records["demographics"] = ConfigRecord({"D1": int(D1[0]), "D2": int(D2[0])})
            assert self._generate_set(1).reset_index()["D1"][0] == D1
            assert self._generate_set(1).reset_index()["D2"][0] == D2

        D1 = np.full(n_samples, self.demographics["D1"])
        D2 = np.full(n_samples, self.demographics["D2"])

        X = D1 + np.random.normal(0, 2, n_samples)
        Y = np.random.normal(D2 - 2 * D1, 1)
        Z = 2 * D2 - np.random.uniform(0, 2, n_samples)

        O1 = np.random.binomial(1, expit(2 * X * Y + 2 * Y + 2 * Z), n_samples)

        train = pd.DataFrame({"X": X, "Y": Y, "Z": Z, "O1": O1})
        x_train = train.drop(["O1"], axis=1).to_numpy()
        y_train = train["O1"].to_numpy()
        train_dataset = IntermediateDataset(x_train,y_train)
        trainloader = DataLoader(train_dataset,batch_size = n_samples)
        # data_loader = DataLoader(data_set, batch_size = len(data_set.dataframe))                                
        #print("TRAINLOADER HERE")                                                                                
        #print(trainloader)                                                                                       
        #print(type(trainloader))                                                                                 
        inputs, targets = next(iter(trainloader))
        O1hat = self.net(inputs).detach().numpy().flatten()
        #print(O1hat)                                                                                             
        #O1hat = self.net(trainloader)[0]                                                                         

        S = np.random.binomial(1, expit(D1 - 10 * (O1 - O1hat) ** 2), n_samples)

        pRS0 = expit(2 * D1)

        R = np.random.binomial(
            1, pRS0 / (pRS0 + np.exp(4 * (1 - S)) * (1 - pRS0)), n_samples
        )



        df = pd.DataFrame(
            {"D1": D1, "D2": D2, "X": X, "Y": Y, "Z": Z, "O1": O1, "S": S, "R": R}
        )

        return df
def client_fn(context: Context) -> Client:
    #context is a dict of necessary information                                                                   
    net = Net().to(DEVICE)
    partition_id = context.node_config["partition-id"]
    #trainloader,valloader,_ = load_datasets(partition_id=partition_id)                                           
    return MyClient(net,context).to_client()






# Flower ClientApp
app = ClientApp(
    client_fn,
)
