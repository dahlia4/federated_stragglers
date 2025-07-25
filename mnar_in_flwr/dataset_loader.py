#Should be obsolete with generated data, but might be nice for real world

from knobs import NUM_CLIENTS, BATCH_SIZE
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset


def load_datasets(partition_id):
    raise("THIS SHOULD NOT BE RUNNING (load datasets)")
    #Load a dataset and a specific partition
    fds = FederatedDataset(dataset="cifar10", partitioners={"train":NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    #Split into train and test
    partition_train_test = partition.train_test_split(test_size=0.2,seed=42)

    #Idk what this is, directly from tutorial, research this later
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )
    def apply_transforms(batch):
        batch["img"]=[pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size = BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"],batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset,batch_size=BATCH_SIZE)

    return trainloader, valloader, testloader
