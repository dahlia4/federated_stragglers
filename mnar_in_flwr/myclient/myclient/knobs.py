import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 50
BATCH_SIZE = 32
NUM_ROUNDS = 1000
NUM_CPUS = 20
MISSING = True
COMPUTE_WEIGHTS = False
