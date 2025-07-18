import socket
import sys
import threading
import json
import pickle
import argparse
import errno
import random

import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from autograd_fast import MLP

from scipy.special import expit
from typing import Any, Dict
from time import sleep

MAX_NUMBER_INCOMING_LISTENS = 10
TRAIN_ROWS_PER_DEVICE = 100

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Device:
    def __init__(
        self,
        device_id,
        server_ip: str = "127.0.0.1",
        port: int = 9987,
        local_ip: str = "127.0.0.1",
        local_port: int = 11234,
        dataset=None,
        use_network=True,
        verbose_mode=False,
    ):
        self.device_id = device_id

        with open("src/satisfaction_model.pkl", "rb") as f:
            self.local_model = pickle.load(f)

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

        self.use_network = use_network

        self._generate_large_train_set(500)

        # Create the devices network connection and send an initial HELO to the server
        if use_network:
            self.network = Network(
                self, server_ip, port, local_ip, local_port, verbose_mode
            )
            self.network.send_hello()

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

    def survey_local(self):
        copy = self.survey_dataset.copy()
        # copy = self._generate_set(1)
        if copy.reset_index()["R"][0] == 0:
            copy["S"] = -1
        # self.dataset = copy[['X', 'Y', 'Z', 'O1']]
        # self.response = copy.reset_index()['R'][0]
        # self.satisfied = copy.reset_index()['S'][0]

        if not self.use_network:
            return copy

        self.network.send_survey_data(copy.iloc[0])

    def train_local(self, msg_id, missing=False):
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

        if not self.use_network:
            return gradient

        self.network.send_gradients(
            gradient, {"num_rows": len(self.dataset), "loss": 1000000, "accuracy": 0},
            msg_id
        )

    def test_local(self, msg_id):
        df = self._generate_set(1).drop(["R", "S", "D1", "D2"], axis=1)

        x_test = df.drop(["O1"], axis=1).to_numpy()
        y_test = df["O1"].to_numpy()

        loss, accuracy = self.local_model.evaluate(x_test, y_test)

        if not self.use_network:
            return loss, accuracy

        self.network.send_eval_metrics(
            {"loss": loss, "accuracy": accuracy, "num_rows": len(self.dataset)},
            msg_id
        )

    def update_local_model(self, global_model):
        # Updates the local model with global_model's parameters
        # Locking to prevent race conditions
        self.model_lock.acquire()
        self.local_model._set_parameters(global_model)
        self.model_lock.release()


class Network:
    def __init__(
        self,
        device: Device,
        server_ip: str = "127.0.0.1",
        port: int = 9987,
        local_ip: str = "127.0.0.1",
        local_port: int = 11234,
        verbose_mode=False,
    ):
        self.server_ip = server_ip
        self.server_port = port
        self.local_ip = local_ip
        self.local_port = local_port
        self.verbose_mode = verbose_mode
        self.listener_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.shutdown = False

        # Bind the socket to the host and port
        self.listener_socket.bind((local_ip, local_port))

        # Store the device for callbacks on messages received from the server
        self.device = device

        # Start a listener for messages initiated by the server
        self.listener_thread = threading.Thread(
            target=self.listener, args=(self.listener_socket, self.verbose_mode)
        )
        self.listener_thread.start()

    def perform_requested_action(self, msg, verbose_mode=False):
        """Performs the action specified by msg on the given device"""

        # print(msg, flush = True)

        match (msg.get("type")):
            case "TRAN":
                if verbose_mode:
                    print(f"Device {self.device.device_id}: Received TRAN", flush=True)
                self.device.train_local(msg.get("msg_id"))
            case "UPDT":
                if verbose_mode:
                    print(f"Device {self.device.device_id}: Received UPDT", flush=True)
                self.device.update_local_model(msg.get("data"))
            case "EVAL":
                if verbose_mode:
                    print(f"Device {self.device.device_id}: Received EVAL", flush=True)
                self.device.test_local(msg.get("msg_id"))
            case "SRVY":
                if verbose_mode:
                    print(f"Device {self.device.device_id}: Received SRVY", flush=True)
                self.device.survey_local()
            case "HALT":
                self.shutdown = True
            case _:
                raise ValueError("Server received invalid message type")

    def listener(self, listener_socket: socket.socket, verbose_mode):
        listener_socket.listen(MAX_NUMBER_INCOMING_LISTENS)
        listener_socket.settimeout(1)

        while not self.shutdown:
            try:
                connection, connection_address = listener_socket.accept()

                received_data = b""

                while True:
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    received_data += chunk

                received_msg = pickle.loads(received_data)

                # print(received_msg, flush =True)

                self.perform_requested_action(received_msg, verbose_mode)

                connection.close()
            except socket.timeout:
                pass

        self.listener_socket.close()

    def send_hello(self):
        response = self._send_to_server("HELO", f"{self.local_ip}:{self.local_port}")
        if self.verbose_mode:
            print(f"Device {self.device.device_id}: Sent HELO", flush=True)

    def send_gradients(
        self, gradients: Dict[str, torch.Tensor], eval_metrics: Dict[str, float], msg_id
    ):
        response = self._send_to_server("GRAD", gradients, eval_metrics, msg_id=msg_id)
        if self.verbose_mode:
            print(f"Device {self.device.device_id}: Sent GRAD", flush=True)

    def send_eval_metrics(self, eval_metrics, msg_id):
        response = self._send_to_server("MTRC", eval_metrics, msg_id=msg_id)
        if self.verbose_mode:
            print(f"Device {self.device.device_id}: Sent MTRC", flush=True)

    def send_survey_data(self, survey_data):
        response = self._send_to_server("SRVY", survey_data)
        if self.verbose_mode:
            print(f"Device {self.device.device_id}: Sent SRVY", flush=True)

    def _send_to_server(
        self, type: str, msg_data: Any, eval_metrics: Dict[str, float] = None, msg_id=None
    ) -> json:
        try:
            socket_to_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except OSError as e:
            if e.errno == errno.EMFILE:
                # The OS has run out of file descriptors. Sleep for 3 seconds and try again.
                print(
                    "WARN: Ran out of file descriptors. Slowing down and retrying.",
                    flush=True,
                )
                sleep(10)
                self._send_to_server(type, msg_data, eval_metrics)
            else:
                raise e

        msg = {"type": type, "data": msg_data, "id": self.device.device_id, "msg_id": msg_id}

        if eval_metrics:
            msg["eval"] = eval_metrics

        encoded_msg = pickle.dumps(msg)

        received_data = b""

        # Connect to the server
        try:
            socket_to_server.connect((self.server_ip, self.server_port))
            socket_to_server.sendall(encoded_msg)

            # while True:
            #     chunk = socket_to_server.recv(4096)
            #     if not chunk:
            #         break
            #     received_data += chunk

            socket_to_server.close()

        except ConnectionRefusedError as e:
            socket_to_server.close()
            print("WARN: Connection refused. Retrying", flush=True)
            print(e)
            sleep(3)
            self._send_to_server(type, msg_data, eval_metrics)

        except ConnectionResetError as e:
            socket_to_server.close()
            print("WARN: Connection Reset. Retrying.", flush=True)
            print(e)
            sleep(3)
            self._send_to_server(type, msg_data, eval_metrics)

        except BrokenPipeError as e:
            socket_to_server.close()
            print("WARN: Broken Pipe. Retrying.", flush=True)
            print(e)
            sleep(3)
            self._send_to_server(type, msg_data, eval_metrics)

        except OSError as e:
            socket_to_server.close()
            print("Hello", e)
            if e.errno == errno.EADDRNOTAVAIL:
                # The OS has run out of ports. Sleep for 3 seconds and try again.
                print(
                    f"WARN: Ran out of ephemeral ports. Slowing down train speed.",
                    flush=True,
                )
                sleep(3)
                self._send_to_server(type, msg_data, eval_metrics)

        # return pickle.loads(received_data)


def parse_command_line_args():
    parser = argparse.ArgumentParser(description="Client Configuration")

    # Server IP argument
    parser.add_argument(
        "-server-ip", type=str, help="The IP address of the server", required=True
    )

    # Server Port argument
    parser.add_argument(
        "-server-port", type=int, help="The port of the server", required=True
    )

    # Verbose argument
    parser.add_argument(
        "-verbose",
        action="store_true",
        help="Whether to run the client in verbose mode",
    )

    # Client IP argument
    parser.add_argument(
        "-client-ip", type=str, help="The IP address of the client", required=True
    )

    # Client Port argument
    parser.add_argument(
        "-client-port",
        type=int,
        help="The port that the client listens on",
        required=True,
    )

    # Device ID argument
    parser.add_argument("-device-id", type=int, help="The device ID", required=True)

    return parser.parse_args()


def main():
    args = parse_command_line_args()

    verbose_mode = args.verbose

    device = Device(
        args.device_id,
        server_ip=args.server_ip,
        port=args.server_port,
        local_ip=args.client_ip,
        local_port=args.client_port,
        verbose_mode=args.verbose,
    )


if __name__ == "__main__":
    main()