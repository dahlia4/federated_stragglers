# from autograd import MLP, negative_loglikelihood
from collections import Counter
import warnings
from autograd_fast import MLP
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from scipy.special import expit
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager, Lock, Semaphore
import threading
import torch
import os
import time
from shadow_recovery import ShadowRecovery

DISABLE_TQDM = True

NUM_EPOCHS = 100
BATCH_SIZE = 16

import pickle

with open("src/satisfaction_model.pkl", "rb") as f:
    satisfaction_model = pickle.load(f)

# satisfaction_model = MLP(3, [16, 1], learning_rate=0.01)

# import pickle

# with open("src/satisfaction_model.pkl", "wb") as f:
#     pickle.dump(satisfaction_model, f)


class Device:
    def __init__(self, device_id, dataset=None):
        self.device_id = device_id
        self.local_model = MLP(3, [16, 1], learning_rate=0.01)

        self.demographics = {}

        if dataset is None:
            dataset = self._generate_set(1)

        self.survey_dataset = dataset[["D1", "D2", "R", "S"]]

        self.response = self.survey_dataset.reset_index(drop=True)["R"][0]
        self.satisfied = self.survey_dataset.reset_index(drop=True)["S"][0]
        self.demographics = {
            "D1": dataset.reset_index()["D1"][0],
            "D2": dataset.reset_index()["D2"][0],
        }

        self.dataset = dataset[["X", "Y", "Z", "O1"]]

        self.model_lock = threading.Lock()

    def _generate_large_train_set(self, num_rows):
        """
        This method generates a larger dataset for training purposes. NOTE: calling this method
        mutates the dataset attribute.
        """
        large_dataset = self._generate_set(num_rows)
        large_dataset = large_dataset[
            (large_dataset["R"] == self.response)
            & (large_dataset["S"] == self.satisfied)
        ]

        assert len(large_dataset) > 0

        self.dataset = large_dataset[["X", "Y", "Z", "O1"]]

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
        O1hat = satisfaction_model.predict(train.drop(["O1"], axis=1).values)

        S = np.random.binomial(1, expit(D1 - 10 * (O1 - O1hat) ** 2), n_samples)

        pRS0 = expit(2 * D1)
        R = np.random.binomial(
            1, pRS0 / (pRS0 + np.exp(4 * (1 - S)) * (1 - pRS0)), n_samples
        )

        df = pd.DataFrame(
            {"D1": D1, "D2": D2, "X": X, "Y": Y, "Z": Z, "O1": O1, "S": S, "R": R}
        )

        return df

    def survey_local(self):
        copy = self.survey_dataset.copy()
        # copy = self._generate_set(1)
        # if copy.reset_index()['R'][0] == 0:
        #     copy['S'] = -1
        # self.dataset = copy[['X', 'Y', 'Z', 'O1']]
        # self.response = copy.reset_index()['R'][0]
        # self.satisfied = copy.reset_index()['S'][0]
        return copy

    def train_local(self, missing=False):
        # if missing:
        #     while True:
        #         df = self._generate_set(1)
        #         if df.reset_index()['R'][0] == self.response and df.reset_index()['S'][0] == self.satisfied:
        #             df = df.drop(['R', 'S', 'D1', 'D2'], axis=1)
        #             break
        # else:
        #     df = self.dataset

        assert len(self.dataset.columns) == 4
        df = self.dataset.sample(n=1)
        self.local_model._zero_grad()
        x_train = df.drop(["O1"], axis=1).to_numpy()
        y_train = df["O1"].to_numpy()

        gradient = self.local_model.train(x_train, y_train)

        return gradient

    def test_local(self):
        df = self._generate_set(1).drop(["R", "S", "D1", "D2"], axis=1)

        x_test = df.drop(["O1"], axis=1).to_numpy()
        y_test = df["O1"].to_numpy()

        loss, accuracy = self.local_model.evaluate(x_test, y_test)

        return loss, accuracy

    def update_local_model(self, global_model):
        # Updates the local model with global_model's parameters
        # Locking to prevent race conditions
        self.model_lock.acquire()
        self.local_model._set_parameters(global_model._get_parameters())
        self.model_lock.release()


def aggregate_gradients(client_grads):
    """Aggregate gradients"""
    return [
        torch.mean(
            torch.stack([client_grad[i] for _, client_grad in client_grads]), dim=0
        )
        for i in range(len(client_grads[0]))
    ]


def run_no_missingness(num_rows):
    global_model = MLP(3, [16, 1], learning_rate=0.01)

    devices = [Device(i) for i in range(num_rows)]

    [device._generate_large_train_set(1000) for device in devices]

    for _ in tqdm(
        range(int(num_rows * NUM_EPOCHS / BATCH_SIZE)), leave=False, disable=None
    ):
        device_sample = random.choices(devices, k=16)

        grads = []

        # Get gradients for each device
        for device in device_sample:
            device.update_local_model(global_model)
            device.local_model._zero_grad()

            grad = device.train_local()

            grads.append([device, grad])

        # Set the aggregated gradients on the global model
        global_model._set_grads(aggregate_gradients(grads))

        global_model.step()

        global_model._zero_grad()

    # Send weights back to devices
    for device in devices:
        device.update_local_model(global_model)

    accuracies = {}

    for device in devices:
        _, accuracy = device.test_local()
        demographic_key = (device.demographics["D1"], device.demographics["D2"])
        accuracies[demographic_key] = accuracies.get(demographic_key, []) + [accuracy]

    assert len(accuracies) > 0

    return {demo : np.mean(accuracy_list) for demo, accuracy_list in accuracies.items()}


def run_missingness(num_rows):
    global_model = MLP(3, [16, 1], learning_rate=0.01)

    filtered_devices = []
    devices = []

    i = 0

    known_demographics = []

    while len(filtered_devices) < num_rows:
        # device = Device(i, dataset=df.iloc[[i]])
        device = Device(i)

        device_survey = device.survey_local().reset_index()
        if device_survey["R"][0] == 1:
            filtered_devices.append(device)
            device._generate_large_train_set(1000)

        known_demographics.append(device_survey)

        devices.append(device)

        i += 1

    known_demographics = pd.concat(known_demographics)

    devices = devices[:500]

    for _ in tqdm(
        range(int(num_rows * NUM_EPOCHS / BATCH_SIZE)), leave=False, disable=None
    ):
        device_sample = random.choices(filtered_devices, k=16)
        grads = []

        # Get gradients for each device
        for device in device_sample:
            device.update_local_model(global_model)
            device.local_model._zero_grad()

            grad = device.train_local(missing=True)

            grads.append([device, grad])

        # Set the aggregated gradients on the global model
        global_model._set_grads(aggregate_gradients(grads))

        global_model.step()

    # Send weights back to devices
    for device in filtered_devices:
        device.update_local_model(global_model)

    accuracies = {}

    for device in devices:
        device.update_local_model(global_model)
        _, accuracy = device.test_local()
        demographic_key = (device.demographics["D1"], device.demographics["D2"])
        accuracies[demographic_key] = accuracies.get(demographic_key, []) + [accuracy]

    assert len(accuracies) > 0

    return {demo : np.mean(accuracy_list) for demo, accuracy_list in accuracies.items()}

def run_missingness_with_correction(num_rows):
    # df = generate_set(20 * num_rows)

    global_model = MLP(3, [16, 1], learning_rate=0.01)

    filtered_devices = []
    devices = []

    i = 0

    known_demographics = []

    while len(filtered_devices) < num_rows:
        # device = Device(i, dataset=df.iloc[[i]])
        device = Device(i)

        device_survey = device.survey_local().reset_index()
        if device_survey["R"][0] == 1:
            filtered_devices.append(device)
            device._generate_large_train_set(1000)

        known_demographics.append(device_survey)

        devices.append(device)

        i += 1

    known_demographics = pd.concat(known_demographics)

    devices = devices[:500]

    recovery = ShadowRecovery("D2", "S", "R", ["D1"], known_demographics)
    recovery._findRoots()

    weights = []

    for device in filtered_devices:
        device_df = pd.DataFrame(
            [
                {
                    "D1": device.demographics["D1"],
                    "D2": device.demographics["D2"],
                    "S": device.satisfied,
                }
            ]
        )
        weight = 1 / float(recovery._propensityScoresRY(device_df)[0])

        weights.append(weight)

    for _ in tqdm(
        range(int(num_rows * NUM_EPOCHS / BATCH_SIZE)), leave=False, disable=None
    ):
        device_sample = random.choices(filtered_devices, k=16, weights=weights)
        grads = []

        # Get gradients for each device
        for device in device_sample:
            device.update_local_model(global_model)
            device.local_model._zero_grad()

            grad = device.train_local(missing=True)

            grads.append([device, grad])

        # Set the aggregated gradients on the global model
        global_model._set_grads(aggregate_gradients(grads))

        # Update the global model
        global_model.step()

    # Send weights back to devices
    for device in filtered_devices:
        device.update_local_model(global_model)

    accuracies = {}

    for device in devices:
        device.update_local_model(global_model)
        _, accuracy = device.test_local()
        demographic_key = (device.demographics["D1"], device.demographics["D2"])
        accuracies[demographic_key] = accuracies.get(demographic_key, []) + [accuracy]

    assert len(accuracies) > 0

    return {demo : np.mean(accuracy_list) for demo, accuracy_list in accuracies.items()}


def create_plot(
    results,
    lock,
    id,
    title
):
    plt.clf()
    lock.acquire()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num_rows_values = list(
            sorted(results.keys()
            )
        )

        demographics = list(results[num_rows_values[0]].keys())

        line_plots = {demographic : [np.mean(results[num_row][demographic]) for num_row in num_rows_values] for demographic in demographics}
        
    lock.release()

    # Customize your title and axis labels
    title_text = f"Accuracy by Demographic Class with {title.title()} Data"
    x_axis_label = "Amount of Devices"
    y_axis_label = "Accuracy"

    # Plot the non-missing and missing mean values
    for demographic, y_values in line_plots.items():
        plt.plot(num_rows_values, y_values, label=demographic, marker="o")

    # Customize the plot
    plt.title(title_text)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()

    # Show the plot
    plt.savefig(f"missingness_impact_{id}_{title}.png")


def run_simulation(
    manager, num_rows, lock, results_non_missing, results_missing, results_missing_with_correction, semaphore
):
    seed = os.getpid() + int(time.time())
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    non_missing_res = run_no_missingness(num_rows)
    missing_res = run_missingness(num_rows)
    missing_res_with_correction = run_missingness_with_correction(num_rows)

    lock.acquire()

    for demographic, accuracy in non_missing_res.items():
        if num_rows not in results_non_missing:
            results_non_missing[num_rows] = manager.dict()
        if demographic not in results_non_missing[num_rows]:
            results_non_missing[num_rows][demographic] = manager.list()
        
        results_non_missing[num_rows][demographic].append(accuracy)

    for demographic, accuracy in missing_res.items():
        if num_rows not in results_missing:
            results_missing[num_rows] = manager.dict()
        if demographic not in results_missing[num_rows]:
            results_missing[num_rows][demographic] = manager.list()
        
        results_missing[num_rows][demographic].append(accuracy)

    for demographic, accuracy in missing_res_with_correction.items():
        if num_rows not in results_missing_with_correction:
            results_missing_with_correction[num_rows] = manager.dict()
        if demographic not in results_missing_with_correction[num_rows]:
            results_missing_with_correction[num_rows][demographic] = manager.list()
        
        results_missing_with_correction[num_rows][demographic].append(accuracy)

    lock.release()
    semaphore.release() 

    create_plot(
        results_non_missing,
        lock,
        id=image_id,
        title = "no_missing"
    )

    create_plot(
        results_missing,
        lock,
        id=image_id,
        title="missing"
    )

    create_plot(
        results_missing_with_correction,
        lock,
        id=image_id,
        title="missing_with_correction"
    )

def print_nested_dict(d):
    res = {}
    for key, value in d.items():
        inner = {}
        for k, v in value.items():
            inner[k] = [x for x in v]
        res[key] = inner

    return res

if __name__ == "__main__":
    num_iters = 30
    MAX_NUM_RUNNING = 64
    lock = Lock()
    semaphore = Semaphore(MAX_NUM_RUNNING)
    image_id = random.randint(1_000_000, 100_000_000)

    print(image_id)

    manager = Manager()
    results_missing = manager.dict()
    results_non_missing = manager.dict()
    results_missing_with_correction = manager.dict()

    num_rows_values = [500 * i for i in range(19, 21)]

    # Call the function with various values for num_rows and run num_iter times
    processes = []
    for num_rows in num_rows_values:
        for _ in range(num_iters):
            p = Process(
                target=run_simulation,
                args=(
                    manager,
                    num_rows,
                    lock,
                    results_non_missing,
                    results_missing,
                    results_missing_with_correction,
                    semaphore,
                ),
            )
            processes.append(p)

    for i, p in enumerate(processes):
        semaphore.acquire()
        p.start()
        print(f"Process {i} out of {len(processes)} started")
        print("no-missing =", print_nested_dict(results_non_missing))
        print("missing =", print_nested_dict(results_missing))
        print("corrected =", print_nested_dict(results_missing_with_correction))

    for p in processes:
        p.join()

    print("no-missing =", print_nested_dict(results_non_missing))
    print("missing =", print_nested_dict(results_missing))
    print("corrected =", print_nested_dict(results_missing_with_correction))
