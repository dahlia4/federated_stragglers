import flwr
from flwr.client import NumPyClient, Client
from flwr.common import Metrics, Context
from knobs import DEVICE
from net import Net, train, test, set_parameters, get_parameters
from dataset_loader import load_datasets
import pandas as pd

class MyClient(NumPyClient):
    """
    To define NumPyClient: init, fit, evaluate, get_parameters
    """
    
    def __init__(self,net,trainloader,valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.properties = {}

        dataset = self._generate_set(1)
        self.demographics = {
            "D1": dataset.reset_index()["D1"][0],
            "D2": dataset.reset_index()["D2"][0],
        }
        
    def get_parameters(self,config):
        #Return parameters the local model has
        return get_parameters(self.net)
    
    def fit(self,parameters,config):
        #Set the local model to have the global parameters
        set_parameters(self.net,parameters)

        #Train model locally for one epoch 
        train(self.net,self.trainloader,1)

        #Return updated parameters, number of examples used for training, dict with "metrics"
        return get_parameters(self.net),len(self.trainloader),{}

    def evaluate(self,parameters,config):
        #Set the local model to have the global parameters
        set_parameters(self.net,parameters)

        #Run our testing function
        loss, accuracy = test(self.net,self.valloader)

        #Return loss, num_examples, and metrics dictionary
        return loss, len(self.valloader), {"accuracy": float(accuracy)}

    def get_properties(self):
        temp_dict = self._generate_set(1).iloc[0].to_dict()
        if temp_dict["R"] == 0:
            temp_dict["S"] = -1
        return temp_dict
    def _generate_set(self, n_samples):
        if len(self.demographics) == 0:
            D1 = np.random.binomial(1, 0.5, 1)
            D2 = np.random.binomial(1, 0.5, 1)
            self.demographics = {"D1": D1, "D2": D2}
            assert self._generate_set(1).reset_index()["D1"][0] == D1
            assert self._generate_set(1).reset_index()["D2"][0] == D2

        D1 = np.full(n_samples, self.demographics["D1"])
        D2 = np.full(n_samples, self.demographics["D2"])

        X = D1 + np.random.normal(0, 2, n_samples)
        Y = np.random.normal(D2 - 2 * D1, 1)
        Z = 2 * D2 - np.random.uniform(0, 2, n_samples)

        O1 = np.random.binomial(1, expit(2 * X * Y + 2 * Y + 2 * Z), n_samples)

        train = pd.DataFrame({"X": X, "Y": Y, "Z": Z, "O1": O1})

        # data_loader = DataLoader(data_set, batch_size = len(data_set.dataframe))                                       
        O1hat = self.local_model.predict(train.drop(["O1"], axis=1).values)

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
    trainloader,valloader,_ = load_datasets(partition_id=partition_id)
    return MyClient(net,trainloader,valloader).to_client()
