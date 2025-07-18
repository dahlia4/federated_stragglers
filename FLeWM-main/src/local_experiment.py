import sys
sys.path.append('./src')

from models.classification_net import ClassificationNN
from models.logistic_regression_net import LogisticRegression
from utils.data_loader import DataLoader
from utils.flewm_optimizer import FlewmOptimizer
from utils.dataframe_dataset import DataFrameDataset
from device import Device
from multiprocessing import Process, Lock, Manager, Semaphore
from sklearn.metrics import accuracy_score
from autograd import MLP
import multiprocessing

from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.special import expit 
import pandas as pd

# def perform_one_simulation(num_devices):
#     # Create the model and the optimizer
#     global_model = ClassificationNN(3)


#     optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01, momentum=0.9)
#     flewm_optimizer = FlewmOptimizer(global_model, optimizer)

#     # Create a list of num_devices devices
#     devices = [Device(i, use_network=False) for i in range(num_devices)]

#     [device.local_model.load_state_dict(global_model.state_dict()) for device in devices]

#     for _ in tqdm(range(250), leave=False):
#         # Train one iteration locally on each device
#         gradients = [device.train_local()[0] for device in random.sample(devices, int(0.5 * len(devices)))]

#         # Update the model with the aggregated gradients
#         flewm_optimizer.step_with_gradients(aggregate_gradients(gradients))

#         [device.local_model.load_state_dict(global_model.state_dict()) for device in devices]

#     # Test the model
#     overall_accuracy = 0.0
    
#     for device in devices:
#         accuracy = device.test_local()
#         overall_accuracy += accuracy

#     # print(overall_accuracy/ len(devices))

#     return overall_accuracy/ len(devices)

# def perform_one_simulation_missingness(num_devices_after_missingness):
#     # Repeat but with missingness
#     global_model = ClassificationNN(3)

#     optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01, momentum=0.9)
#     flewm_optimizer = FlewmOptimizer(global_model, optimizer)

#     # Create a list of num_devices devices
#     devices = []
#     filtered_devices = []
#     i = 0
#     while len(filtered_devices) < num_devices_after_missingness:
#         device = Device(i, use_network=False)

#         if device.survey_local()['R'] == 1:
#             filtered_devices.append(device)

#         devices.append(device)

#         i += 1

#     devices = devices[:num_devices_after_missingness]

#     [device.local_model.load_state_dict(global_model.state_dict()) for device in filtered_devices]

#     # devices = [Device(i, use_network=False) for i in range(10 * num_devices_after_missingness)]

#     # filtered_devices = [device for device in devices if device.generator.generate_survey_data(global_model)['R'] == 1]

#     for _ in tqdm(range(250), leave=False):
#         # Train one iteration locally on each device
#         gradients = [device.train_local()[0] for device in random.sample(filtered_devices, int(0.5 * len(filtered_devices)))]

#         # Update the model with the aggregated gradients
#         flewm_optimizer.step_with_gradients(aggregate_gradients(gradients))

#         [device.local_model.load_state_dict(global_model.state_dict()) for device in filtered_devices]

#     # Test the model
#     overall_accuracy = 0.0
    
#     for device in devices:
#         accuracy = device.test_local()
#         overall_accuracy += accuracy

#     # print(overall_accuracy/ len(filtered_devices))

#     return overall_accuracy/ len(devices)

def perform_one_simulation(num_devices):
    # df = generate_set(num_devices)

    # # Create the model and the optimizer
    # global_model = LogisticRegression(3)

    # optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01)
    # flewm_optimizer = FlewmOptimizer(global_model, optimizer)

    # # Create a list of num_devices devices
    # devices = [Device(i, use_network=False, dataset=df.iloc[[i]]) for i in range(num_devices)]

    # [device.update_weights(global_model.state_dict()) for device in devices] # simulate sending the weights to the model

    # for _ in tqdm(range(50), leave=False):
    #     # Train one iteration locally on each device
    #     # gradients = [device.train_local()[0] for device in random.sample(devices, 16)]
    #     random.shuffle(devices)
    #     for i in range(0, len(devices), 16):
    #         gradients = [device.train_local()[0] for device in devices[i:i+16]]

    #     # Update the model with the aggregated gradients
    #         flewm_optimizer.step_with_gradients(aggregate_gradients(gradients))

    #         [device.update_weights(global_model.state_dict()) for device in devices]

    # # Test the model
        
    # df = generate_set(500)
    # df = df.drop(['D1', 'D2', 'R', 'S'], axis = 1)
    # test_set = DataFrameDataset(df, labels='O1')
    # no_missing_preds = global_model.predict(test_set.features)

    # return accuracy_score(df['O1'], no_missing_preds)

    df = generate_set(num_rows).drop(['R', 'S', 'D1', 'D2'], axis=1)
    X = df.drop(['O1'], axis=1)
    y = df['O1']

    global_model = MLP(3, [1], learning_rate=0.01, dropout_proba=0.3)

    devices = [Device(i, use_network=False, dataset=df.iloc[[i]]) for i in range(num_devices)]

    for _ in tqdm(range(100), leave=False):
        random.shuffle(devices)
        for i in range(0, len(devices), 16):
            grads = []

            # Get gradients for each device
            for device in devices[i:i+16]:

                device.local_model._zero_grad()
                # preds = device(X.iloc[[i]].to_numpy()[0])

                # negative_loglikelihood(y.iloc[[i]].to_numpy()[0], preds).backward()

                # grads.append(device._get_grads())

                grad, _, _= device.train_local()

                grads.append(grad)

            # Set the aggregated gradients on the global model
            global_model._set_grads(aggregate_gradients(grads))

            # Update the global model
            for p in global_model.parameters():
                p.data -= global_model.learning_rate * p.grad

            global_model._zero_grad()

            # Send weights back to devices
            for device in devices:
                device.local_model._set_parameters(global_model._get_parameters())

    # Test global model
    test = generate_set(500).drop(['R', 'S', 'D1', 'D2'], axis=1)
    x_test = test.drop(['O1'], axis=1)
    y_test = test['O1']

    loss, accuracy = global_model.evaluate(x_test.to_numpy(), y_test.to_numpy())
    # accuracy = np.mean(preds == y_test.to_numpy())

    return accuracy

def perform_one_simulation_missingness(num_devices):
    # # Repeat but with missingness
    # df = generate_set(num_devices * 20)
    # global_model = LogisticRegression(3)

    # optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01)
    # flewm_optimizer = FlewmOptimizer(global_model, optimizer)

    # # Create a list of num_devices devices
    # devices = []
    # filtered_devices = []
    # i = 0
    # while len(filtered_devices) < num_devices:
    #     device_dataset = df.iloc[[i]]
    #     device = Device(i, use_network=False, dataset=device_dataset)

    #     device.update_weights(global_model.state_dict())

    #     if device_dataset.reset_index()['R'][0] == 1:
    #         filtered_devices.append(device)

    #     devices.append(device)

    #     i += 1

    # devices = devices[:num_devices]

    # [device.update_weights(global_model.state_dict()) for device in filtered_devices]

    # for _ in tqdm(range(50), leave=False):
    #     # Train one iteration locally on each device
    #     # gradients = [device.train_local() for device in random.choices(filtered_devices, weights=filtered_weights, k = int(0.5 * len(filtered_devices)))]
    #     # gradients = [device.train_local() for device in random.sample(filtered_devices, k = 16)]
    #     # grads = [gradient[0] for gradient in gradients]
    #     random.shuffle(filtered_devices)
    #     for i in range(0, len(filtered_devices), 16):
    #         gradients = [device.train_local()[0] for device in filtered_devices[i:i+16]]

    #         # Update the model with the aggregated gradients
    #         flewm_optimizer.step_with_gradients(aggregate_gradients(gradients))

    #         [device.update_weights(global_model.state_dict()) for device in filtered_devices]

    # # Test the model
    # [device.update_weights(global_model.state_dict()) for device in devices]
    
    # df = generate_set(500)
    # df = df.drop(['D1', 'D2', 'R', 'S'], axis = 1)
    # test_set = DataFrameDataset(df, labels='O1')
    # missing_preds = global_model.predict(test_set.features)

    # return accuracy_score(df['O1'], missing_preds)
    df = generate_set(num_devices * 20)

    devices = []
    filtered_devices = []
    i = 0

    global_model = MLP(3, [1], learning_rate=0.01, dropout_proba=0.3)

    while len(filtered_devices) < num_devices:
        device_dataset = df.iloc[[i]]
        device = Device(i, use_network=False, dataset=device_dataset)

        device.local_model._set_parameters(global_model._get_parameters())

        if device_dataset.reset_index()['R'][0] == 1:
            filtered_devices.append(device)

        devices.append(device)

        i += 1

    for _ in tqdm(range(100), leave=False):
        random.shuffle(devices)
        for i in range(0, len(devices), 16):
            grads = []

            # Get gradients for each device
            for device in devices[i:i+16]:

                device.local_model._zero_grad()
                # preds = device(X.iloc[[i]].to_numpy()[0])

                # negative_loglikelihood(y.iloc[[i]].to_numpy()[0], preds).backward()

                # grads.append(device._get_grads())

                grad, _, _ = device.train_local()

                grads.append(grad)

            # Set the aggregated gradients on the global model
            global_model._set_grads(aggregate_gradients(grads))

            # Update the global model
            for p in global_model.parameters():
                p.data -= global_model.learning_rate * p.grad

            global_model._zero_grad()

            # Send weights back to devices
            for device in filtered_devices:
                device.local_model._set_parameters(global_model._get_parameters())

    # Test global model
    test = generate_set(500).drop(['R', 'S', 'D1', 'D2'], axis=1)
    x_test = test.drop(['O1'], axis=1)
    y_test = test['O1']

    loss, accuracy = global_model.evaluate(x_test.to_numpy(), y_test.to_numpy())
    # accuracy = np.mean(preds == y_test.to_numpy())

    return accuracy


def run_simulation(num_rows, lock, results_non_missing, results_missing, semaphore):
    # non_missing_res = perform_one_simulation(num_rows)
    missing_res = perform_one_simulation_missingness(num_rows)

    lock.acquire()
    # results_non_missing[num_rows] = results_non_missing.get(num_rows, []) + [non_missing_res]
    results_missing[num_rows] = results_missing.get(num_rows, []) + [missing_res]
    semaphore.release()
    lock.release()

def aggregate_gradients(client_grads):
    """Aggregate gradients"""

    # client_grads = [grad for grad in client_grads]

    return np.mean(client_grads, axis=0)

def create_plot(results_non_missing, results_missing, lock, id):
    plt.clf()
    lock.acquire()
    num_rows_values = list(sorted(results_non_missing.keys()))
    non_missing_y_values = [np.mean(results_non_missing[num_rows]) for num_rows in num_rows_values]
    missing_y_values = [np.mean(results_missing[num_rows]) for num_rows in num_rows_values]
    lock.release()

    # Customize your title and axis labels
    title_text = "Impact of MNAR Data on Accuracy"
    x_axis_label = "Amount of Training Data "
    y_axis_label = "Accuracy"

    # Create x values based on the number of num_rows_values
    x_values = range(len(num_rows_values))

    # Plot the non-missing and missing mean values
    plt.plot(num_rows_values, non_missing_y_values, label="No Missing Data", marker='o')
    plt.plot(num_rows_values, missing_y_values, label="MNAR Data", marker='o')

    # Customize the plot
    plt.title(title_text)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()

    # Show the plot
    plt.savefig(f"missingness_impact_{id}.png")

def generate_set(n_samples, verbose = False):
    satisfaction_model = LogisticRegression(3)
    D1 = np.random.binomial(1, 0.5, n_samples)
    D2 = np.random.binomial(1, 0.5, n_samples)

    X = D1 + np.random.normal(0, 2, n_samples)
    Y = np.random.normal(D2 - 2*D1, 1)
    Z = 2 * D2 - np.random.uniform(0, 2, n_samples)
    
    O1 = np.random.binomial(1, expit(2*X + 2*Y + 2*Z), n_samples)

    train = pd.DataFrame({'X' : X, 'Y' : Y, 'Z' : Z, 'O1' : O1})

    data_set = DataFrameDataset(train, labels = 'O1')
    #data_loader = DataLoader(data_set, batch_size = len(data_set.dataframe))
    O1hat = satisfaction_model.predict(data_set.features)

    S = np.random.binomial(1,  expit(D1 - 8*(O1-O1hat)**2), n_samples)
        
    pRS0 = expit(2*D1)
    R = np.random.binomial(1, pRS0/(pRS0 + np.exp(6*(1-S))*(1 - pRS0)), n_samples)

    df = pd.DataFrame({'D1' : D1, 'D2' : D2, 'X' : X, 'Y' : Y, 'Z' : Z, 'O1' : O1, 'S' : S, 'R' : R})

    if verbose:
        print("R", np.mean(R))
        print("O1", np.mean(O1), np.mean(O1hat))
        print("S", np.mean(S))

    return df

if __name__ == "__main__":
    num_iters = 10
    MAX_NUM_RUNNING = 64
    lock = Lock()
    semaphore = Semaphore(MAX_NUM_RUNNING)
    image_id = random.randint(0, 100_000_000)

    manager = Manager()
    results_missing = manager.dict()
    results_non_missing = manager.dict()

    num_rows_values = [100 * i for i in range(5, 6)] 

    # Call the function with various values for num_rows and run num_iter times
    processes = []
    for num_rows in num_rows_values:
        for _ in range(num_iters):
            p = Process(target=run_simulation, args=(num_rows, lock, results_non_missing, results_missing, semaphore))
            processes.append(p)

    for i, p in enumerate(processes):
        semaphore.acquire()
        p.start()
        print(f"Process {i} out of {len(processes)} finished")
        create_plot(results_non_missing, results_missing, lock, id=image_id)

    for p in processes:
        p.join()

    print(results_missing)
    print(results_non_missing)

    create_plot(results_non_missing, results_missing, lock, image_id)