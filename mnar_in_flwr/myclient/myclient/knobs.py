import torch
from .timeout_straggler import timeout

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#NUM_CLIENTS = 50
BATCH_SIZE = 32
NUM_ROUNDS = 500

MISSING = False
COMPUTE_WEIGHTS = False

STRAGGLERS = True
STRAGGLER_TIMEOUT = timeout
