import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 50
BATCH_SIZE = 32
NUM_ROUNDS = 50
NUM_CPUS = 1
