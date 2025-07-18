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

class Device:
    def __init__(self, device_id, dataset=None):
        self.device_id = device_id
        self.local_model = MLP(3, [16, 1], learning_rate=0.01)

        self.demographics = {}

        self.response = None
        self.satisfied = None

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
        while True:
            large_dataset = self._generate_set(num_rows)
            large_dataset = large_dataset[
                (large_dataset["R"] == self.response)
                & (large_dataset["S"] == self.satisfied)
            ]

            if len(large_dataset) > 0:
                break

        self.dataset = large_dataset[["X", "Y", "Z", "O1"]]

    def _generate_set(self, n_samples, previous_satisfaction = None):
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

        O1hat = self.local_model.predict(train.drop(["O1"], axis=1).values)

        if previous_satisfaction is None:
            S = np.random.binomial(1, expit(D1 - 10 * (O1 - O1hat) ** 2), n_samples)
        else:
            if previous_satisfaction:
                S = np.random.binomial(1, expit(D1 - 10 * (O1 - O1hat) ** 2), n_samples)
            else:
                S = np.random.binomial(1, expit(D1 - 10 * (O1 - O1hat) ** 2 - 2), n_samples)
            

        pRS0 = expit(2 * D1)
        R = np.random.binomial(
            1, pRS0 / (pRS0 + np.exp(4 * (1 - S)) * (1 - pRS0)), n_samples
        )

        df = pd.DataFrame(
            {"D1": D1, "D2": D2, "X": X, "Y": Y, "Z": Z, "O1": O1, "S": S, "R": R}
        )

        return df

    def survey_local(self):
        copy = self._generate_set(1, previous_satisfaction=self.satisfied)
        if copy.reset_index()['R'][0] == 0:
            copy['S'] = -1
        self.dataset = copy[['X', 'Y', 'Z', 'O1']]
        self.response = copy.reset_index()['R'][0]
        self.satisfied = copy.reset_index()['S'][0]
        return copy

    def train_local(self):
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

def accuracy_by_demographic(participating_devices, non_participating_devices):
    res = {}

    for device in participating_devices + non_participating_devices:
        if (device.demographics["D1"], device.demographics["D2"]) not in res:
            res[(device.demographics["D1"], device.demographics["D2"])] = []

        res[(device.demographics["D1"], device.demographics["D2"])].append(device.test_local()[1])

    for demo, accuracies in res.items():
        res[demo] = np.mean(accuracies)

    return res

def run_missingness_response_experiment(num_devices, num_epochs):
    with open("src/satisfaction_model.pkl", "rb") as f:
        global_model = pickle.load(f)

    participating_devices = [Device(i) for i in range(num_devices)]
    non_participating_devices = []

    # [device._generate_large_train_set(1) for device in participating_devices + non_participating_devices]

    from collections import Counter

    counts = Counter([(device.demographics["D1"], device.demographics["D2"]) for device in participating_devices])

    def survey(participating, non_participating):
        new_participating_devices = []
        new_non_participating_devices = []

        for device in participating + non_participating:
            device.update_local_model(global_model)
            device_survey = device.survey_local().reset_index()

            if device_survey["R"][0] == 1:
                new_participating_devices.append(device)
            else:
                new_non_participating_devices.append(device)

        return new_participating_devices, new_non_participating_devices

    results = []

    for i in tqdm(range(num_epochs), leave=False, disable=None):
        participating_devices, non_participating_devices = survey(participating_devices, non_participating_devices)

        [device._generate_large_train_set(1) for device in participating_devices]

        for _ in tqdm(
            range(300), leave=False, disable=None
        ):
            device_sample = random.choices(participating_devices, k=16)
            grads = []

            # Get gradients for each device
            for device in device_sample:
                device.update_local_model(global_model)
                device.local_model._zero_grad()

                grad = device.train_local()

                grads.append([device, grad])

            # Set the aggregated gradients on the global model
            global_model._set_grads(aggregate_gradients(grads))

            # Update the global model
            global_model.step()

        print(global_model._get_parameters())

        # Measure participation
        participation = Counter([(device.demographics["D1"], device.demographics["D2"]) for device in participating_devices])

        results.append({demo : participation[demo] / counts[demo] for demo in sorted(counts.keys())})

        print(f"Global Model Acheived Accuracy with Missing Data after epoch {i}: {[(demo, accuracy) for (demo, accuracy) in sorted(accuracy_by_demographic(participating_devices, non_participating_devices).items())]}, total {np.mean([device.test_local()[1] for device in participating_devices + non_participating_devices])}")

    print(f"Global Model Acheived Accuracy with Missing Data: {np.mean([device.test_local()[1] for device in participating_devices + non_participating_devices])}")

    print(results)

    return results

def run_missingness_correction_response_experiment(num_devices, num_epochs):
    with open("src/satisfaction_model.pkl", "rb") as f:
        global_model = pickle.load(f)

    participating_devices = [Device(i) for i in range(num_devices)]
    non_participating_devices = []

    # [device._generate_large_train_set(1) for device in participating_devices + non_participating_devices]

    from collections import Counter

    counts = Counter([(device.demographics["D1"], device.demographics["D2"]) for device in participating_devices])

    def survey(participating, non_participating):
        new_participating_devices = []
        new_non_participating_devices = []

        known_demographics = []

        for device in participating + non_participating:
            device.update_local_model(global_model)
            device_survey = device.survey_local().reset_index()

            if device_survey["R"][0] == 1:
                new_participating_devices.append(device)
            else:
                new_non_participating_devices.append(device)

            known_demographics.append(device_survey)

        return new_participating_devices, new_non_participating_devices, known_demographics

    results = []

    for i in tqdm(range(num_epochs), leave=False, disable=None):
        participating_devices, non_participating_devices, known_demographics = survey(participating_devices, non_participating_devices)

        [device._generate_large_train_set(1) for device in participating_devices]

        recovery = ShadowRecovery("D2", "S", "R", ["D1"], pd.concat(known_demographics))
        recovery._findRoots()

        weights = []

        for device in participating_devices:
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
            range(300), leave=False, disable=None
        ):
            device_sample = random.choices(participating_devices, k=16)
            grads = []

            # Get gradients for each device
            for device in device_sample:
                device.update_local_model(global_model)
                device.local_model._zero_grad()

                grad = device.train_local()

                grads.append([device, grad])

            # Set the aggregated gradients on the global model
            global_model._set_grads(aggregate_gradients(grads))

            # Update the global model
            global_model.step()

        print(global_model._get_parameters())

        # Measure participation
        participation = Counter([(device.demographics["D1"], device.demographics["D2"]) for device in participating_devices])

        results.append({demo : participation[demo] / counts[demo] for demo in sorted(counts.keys())})

        print(f"Global Model Acheived Accuracy with Missing Data after epoch {i}: {[(demo, accuracy) for (demo, accuracy) in sorted(accuracy_by_demographic(participating_devices, non_participating_devices).items())]}, total {np.mean([device.test_local()[1] for device in participating_devices + non_participating_devices])}")

    print(f"Global Model Acheived Accuracy with Missing Data: {np.mean([device.test_local()[1] for device in participating_devices + non_participating_devices])}")

    print(results)

    return results

# def run_missingness_correction_response_experiment(num_devices, num_epochs):
#     with open("src/satisfaction_model.pkl", "rb") as f:
#         global_model = pickle.load(f)

#     participating_devices = [Device(i) for i in range(num_devices)]
#     non_participating_devices = []

#     [device._generate_large_train_set(1) for device in participating_devices + non_participating_devices]
    
#     known_demographics = []

#     from collections import Counter

#     counts = Counter([(device.demographics["D1"], device.demographics["D2"]) for device in participating_devices])

#     def survey():
#         nonlocal known_demographics
#         known_demographics = []
#         new_participating_devices = []
#         new_non_participating_devices = []
#         for device in participating_devices + non_participating_devices:
#             device.update_local_model(global_model)
#             device_survey = device.survey_local().reset_index()

#             if device_survey["R"][0] == 1:
#                 new_participating_devices.append(device)
#             else:
#                 new_non_participating_devices.append(device)

#             known_demographics.append(device_survey)

#         print(len(new_participating_devices), len(new_non_participating_devices))

#         return new_participating_devices, new_non_participating_devices

#     results = []

#     for i in tqdm(range(num_epochs), leave=False, disable=None):
#         participating_devices, non_participating_devices = survey()

#         recovery = ShadowRecovery("D2", "S", "R", ["D1"], pd.concat(known_demographics))
#         recovery._findRoots()

#         weights = []

#         for device in participating_devices:
#             device_df = pd.DataFrame(
#                 [
#                     {
#                         "D1": device.demographics["D1"],
#                         "D2": device.demographics["D2"],
#                         "S": device.satisfied,
#                     }
#                 ]
#             )
#             weight = 1 / float(recovery._propensityScoresRY(device_df)[0])

#             weights.append(weight)

#         for _ in tqdm(
#             range(300), leave=False, disable=None
#         ):
#             device_sample = random.choices(participating_devices, k=16, weights=weights)
#             grads = []

#             # Get gradients for each device
#             for device in device_sample:
#                 device.update_local_model(global_model)
#                 device.local_model._zero_grad()

#                 grad = device.train_local(missing=True)

#                 grads.append([device, grad])

#             # Set the aggregated gradients on the global model
#             global_model._set_grads(aggregate_gradients(grads))

#             # Update the global model
#             global_model.step()

#         # Measure participation
#         participation = Counter([(device.demographics["D1"], device.demographics["D2"]) for device in participating_devices])

#         results.append({demo : participation[demo] / counts[demo] for demo in sorted(counts.keys())})

#     print(f"Global Model Acheived Accuracy with Correction: {np.mean([device.test_local()[1] for device in participating_devices + non_participating_devices])}")

#     print(results)

#     return results

def create_plot(
    results,
    id,
    title
):
    plt.clf()

    # Customize your title and axis labels
    title_text = f"Participation by Demographic Class with {title.title()}"
    x_axis_label = "Number of Epochs"
    y_axis_label = "Participation"

    # Plot the non-missing and missing mean values
    for demographic in sorted(results.keys()):
        plt.plot([1+i for i in range(len(results[demographic]))], results[demographic], label=demographic, marker="o")

    # Customize the plot
    plt.title(title_text)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()

    # Show the plot
    plt.savefig(f"participation_by_epoch_{id}_{title}.png")

if __name__ == "__main__":
    id = random.randint(1_000_000, 9_999_999)

    print(id)

    missing_results = run_missingness_response_experiment(1000, 20)
    corrected_results = run_missingness_correction_response_experiment(1000, 20)
    
    aggregated_missing = {}
    aggregated_corrected = {}

    for result in missing_results:
        for demo, participation in result.items():
            if demo not in aggregated_missing:
                aggregated_missing[demo] = []
            aggregated_missing[demo].append(participation)
    
    for result in corrected_results:
        for demo, participation in result.items():
            if demo not in aggregated_corrected:
                aggregated_corrected[demo] = []
            aggregated_corrected[demo].append(participation)

    print(aggregated_missing)
    print(aggregated_corrected)

    create_plot(aggregated_missing, id, "missing data")
    create_plot(aggregated_corrected, id, "missing data and correction")