import os
#Just suppressing unnecessary messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["RAY_DEDUP_LOGS"] = "0"

import flwr
from client import client_fn
from server import server
from knobs import NUM_CLIENTS, DEVICE, NUM_CPUS
from flwr.client import ClientApp
from flwr.simulation import run_simulation


if __name__ == "__main__":
    #Define client app
    client = ClientApp(client_fn=client_fn)
    backend_config = {"client_resources": {"num_cpus": NUM_CPUS, "num_gpus": 0.0}}
    if DEVICE == "cuda":
        print("CHOSE CUDA")
        backend_config = {"client_resources": {"num_cpus": NUM_CPUS, "num_gpus": 1.0}}
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )
