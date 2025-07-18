import flwr
from flwr.client import NumPyClient, Client
from flwr.common import Metrics, Context
from knobs import DEVICE
from net import Net, train, test, set_parameters, get_parameters
from dataset_loader import load_datasets


class MyClient(NumPyClient):
    """
    To define NumPyClient: init, fit, evaluate, get_parameters
    """
    
    def __init__(self,net,trainloader,valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

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

def client_fn(context: Context) -> Client:
    #context is a dict of necessary information
    net = Net().to(DEVICE)
    partition_id = context.node_config["partition-id"]
    trainloader,valloader,_ = load_datasets(partition_id=partition_id)
    return MyClient(net,trainloader,valloader).to_client()
