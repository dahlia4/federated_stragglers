from typing import List, Tuple
from shadow_recovery import ShadowRecovery

import argparse
import socket
import threading
import torch
import pickle
import random
import errno
import pandas as pd
import numpy as np


shutdown = False
verbose_mode = False


class Server:
    def __init__(
        self, missing_mode, host="localhost", port=9987, max_queued_connections=100
    ):
        self.missing_mode = missing_mode

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))

        self.max_queued_connections = max_queued_connections

        with open("src/satisfaction_model.pkl", "rb") as f:
            self.global_model = pickle.load(f)

        self.device_addresses = {}
        self.participating_devices = []
        self.missing_devices = []

        self.client_grads = []
        self.client_eval_metrics = []

        self.survey_responses = {}

        self.survey_responses = {}

        self.msg_id = 0

        self.shadow_recovery = None

        self.lock = threading.Lock()
        self.train_cv = threading.Condition(self.lock)

    def serve(self):
        self.server_socket.listen(self.max_queued_connections)
        self.server_socket.settimeout(1)

        while not shutdown:
            try:
                client_socket, client_address = self.server_socket.accept()

                client_thread = threading.Thread(
                    target=self.serve_request, args=(client_socket, client_address)
                )
                client_thread.start()
            except socket.timeout:
                pass  # Continue the loop if accept times out

    def serve_request(self, client_socket: socket.socket, client_address):
        # print(f"Connection from {client_address}")

        received_data = b""

        # received_data = client_socket.recv(2**30)

        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            received_data += chunk

        message = pickle.loads(received_data)

        # print(message)

        self.perform_requested_action(client_socket, message)

        # Close the client socket when done
        client_socket.close()

    def perform_requested_action(self, client_socket, msg):
        # print(msg)

        client_id = msg.get("id")
        match msg.get("type"):
            case "HELO":
                addr, port = msg["data"].split(":")

                self.lock.acquire()
                self.device_addresses[client_id] = (addr, int(port))
                self.lock.release()

                self.send_global_weights([(addr, int(port))])
            case "GRAD":
                self.lock.acquire()
                if msg.get("msg_id") == self.msg_id:
                    self.client_grads.append(msg.get("data"))
                    self.client_eval_metrics.append(msg.get("eval"))
                    self.train_cv.notify()
                self.lock.release()
            case "MTRC":
                self.lock.acquire()
                if msg.get("msg_id") == self.msg_id:
                    self.client_eval_metrics.append(msg.get("data"))
                    self.train_cv.notify()
                self.lock.release()
            case "SRVY":
                self.lock.acquire()
                self.survey_responses[client_id] = msg.get("data")[
                    ["R", "S", "D1", "D2"]
                ]
                self.train_cv.notify()
                self.lock.release()
            case _:
                raise ValueError("Server received invalid message type")

        ack = pickle.dumps({"type": "ACKN", "data": None})

        client_socket.sendall(ack)

    def send_survey_notification(self, clients: List[Tuple[str, int]]):
        msg = pickle.dumps({"type": "SRVY", "data": None})

        self._send_clients_message(clients, msg)

    def send_global_weights(self, clients: List[Tuple[str, int]]):
        """Send the current global weights to each client in clients"""

        self.lock.acquire()
        weights = self.global_model._get_parameters()
        self.lock.release()

        msg = pickle.dumps({"type": "UPDT", "data": weights})
        self._send_clients_message(clients, msg)

    def send_train_notification(self, clients):
        """Send training notification to selected clients"""
        self.lock.acquire()
        msg = pickle.dumps({"type": "TRAN", "data": None, "msg_id": self.msg_id})
        self.lock.release()

        self._send_clients_message(clients, msg)

    def send_eval_notification(
        self, clients: List[Tuple[str, int]], dataset_to_use: str
    ):
        """Send training notification to selected clients"""

        self.lock.acquire()
        msg = pickle.dumps({"type": "EVAL", "data": dataset_to_use, "msg_id": self.msg_id})
        self.lock.release()

        self._send_clients_message(clients, msg)

    def send_shutdown(self):
        msg = pickle.dumps({"type": "HALT"})

        self._send_clients_message(self.device_addresses.values(), msg)

    def _send_clients_message(self, clients: List[Tuple[str, int]], msg):
        msgs = pickle.loads(msg)
        for client in clients:
            # if verbose_mode:
            #     print(f"Sent {msgs.get('type')} to Device {client}", flush = True)

            socket_to_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            try:
                socket_to_client.connect(client)
                socket_to_client.sendall(msg)
            except OSError as e:
                if e.errno == errno.EADDRNOTAVAIL:
                    # Run out of Ephemeral ports. As a bandaid, we sleep for 3 seconds and
                    # call this method again.
                    print(
                        f"WARN: Ran out of ephemeral ports. Slowing down train speed.",
                        flush=True,
                    )
                    sleep(3)
                    self._send_clients_message([client], msg)
            except Exception as e:
                print(self.device_manager.participating_clients, flush=True)
                print(self.device_manager.non_participating_clients, flush=True)
                print(f"ERROR CONNECTING TO CLIENT\n {e}", flush=True)
            finally:
                socket_to_client.close()

    def survey_devices(self):
        new_participating_devices = []
        new_missing_devices = []

        # Send survey notifications to all devices
        self.lock.acquire()
        self.lock.release()
        clients_to_survey = list(self.device_addresses.values())

        self.send_survey_notification(clients_to_survey)

        # Wait for all devices to respond
        self.lock.acquire()
        finished = self.train_cv.wait_for(
            lambda: len(self.survey_responses) == len(clients_to_survey), timeout=60
        )

        if not finished:
            # Some devices did not respond
            pass  # TODO: Mark devices as non-responsive

        self.lock.release()

        # Update the list of participating and missing devices
        for device_id, device_survey in self.survey_responses.items():
            if self.missing_mode != "no-missing":
                if device_survey["R"] == 1:
                    new_participating_devices.append(device_id)
                else:
                    new_missing_devices.append(device_id)
            else:
                new_participating_devices.append(device_id)

        self.lock.acquire()
        self.participating_devices = new_participating_devices
        self.missing_devices = new_missing_devices
        self.lock.release()

    def train(self, num_steps, train_proportion, min_clients):
        print(f"Starting Training.", flush=True)

        for step in range(num_steps):
            num_train = 0

            self.lock.acquire()
            num_clients = len(self.participating_devices)

            if self.missing_mode == "no-missing":
                num_clients += len(self.missing_devices)
            self.lock.release()

            if num_clients < min_clients:
                num_train = num_clients
            elif num_clients * train_proportion < min_clients:
                num_train = min_clients
            else:
                num_train = int(num_clients * train_proportion)

            # Select num_train random clients
            self.lock.acquire()
            clients_to_train = self.participating_devices

            if self.missing_mode == "no-missing":
                clients_to_train += self.missing_devices

            self.msg_id = random.randint(0, 2 ** 32)

            self.lock.release()

            weights = self.compute_weights(clients_to_train)
            clients_to_train = random.choices(
                clients_to_train, k=num_train, weights=weights
            )

            # Notify the clients to start training
            self.send_global_weights([self.device_addresses[client] for client in clients_to_train])
            self.send_train_notification(
                [self.device_addresses[client] for client in clients_to_train]
            )

            # Wait for all clients to return their gradients
            self.lock.acquire()
            finished = self.train_cv.wait_for(
                lambda: len(self.client_grads) == len(clients_to_train), timeout=60
            )

            # Remove unresponsive clients here
            if not finished:
                # Some clients did not respond
                pass

            # Compute the aggregated gradient and set it on the global model
            gradients = self.aggregate_gradients()

            # Update the global model
            # Set the aggregated gradients on the global model
            self.global_model._set_grads(gradients)

            self.global_model.step()

            self.lock.release()

            print(f"Train Iteration {step} of {num_steps} complete", flush=True)

            # Clear the state
            self.client_grads = []
            self.client_eval_metrics = []

        self.send_global_weights(self.device_addresses.values())

    def test(self, num_test: int):
        # assert len(self.device_manager.participating_clients) > 0

        num_test = min(
            num_test, len(self.participating_devices) + len(self.missing_devices)
        )

        print("Starting Testing", flush=True)

        self.lock.acquire()
        clients_to_test = random.sample(
            self.participating_devices + self.missing_devices, k=num_test
        )
        self.msg_id = random.randint(0, 2 ** 32)
        self.lock.release()

        # Notify the clients to start testing here
        self.send_eval_notification(
            [self.device_addresses[client] for client in clients_to_test], "test"
        )

        # Wait for all clients to return their metrics
        self.lock.acquire()

        finished = self.train_cv.wait_for(
            lambda: len(self.client_eval_metrics) == num_test, timeout=60
        )

        if not finished:
            # Some clients did not respond
            pass

        metrics = self.aggregate_eval_metrics()

        print(
            f"Global Model Acheived {metrics[1]}% accuracy on the test data with {metrics[0]} loss"
        )

        self.lock.release()

        self.client_eval_metrics = []

        return metrics

    def compute_weights(self, clients):
        """Returns a list of weights for each client based on propensity scores."""
        if self.missing_mode == "missing" or self.missing_mode == "no-missing":
            return [1] * len(clients)

        if self.shadow_recovery is None:
            self.shadow_recovery = ShadowRecovery(
                "D2",
                "S",
                "R",
                ["D1"],
                pd.DataFrame(list(self.survey_responses.values())),
            )
            self.shadow_recovery._findRoots()

        res = []

        for id in clients:
            score = self.shadow_recovery._propensityScoresRY(self.survey_responses[id])
            res.append(1 / score)

        return res

    def aggregate_gradients(self):
        return [
            torch.mean(
                torch.stack([client_grad[i] for client_grad in self.client_grads]),
                dim=0,
            )
            for i in range(len(self.client_grads[0]))
        ]

    def aggregate_eval_metrics(self):
        losses = [mapping["loss"] for mapping in self.client_eval_metrics]
        accuracies = [mapping["accuracy"] for mapping in self.client_eval_metrics]

        return np.mean(losses), np.mean(accuracies)

    # def send_global_weights(self):
    #     for device in self.participating_devices + self.missing_devices:
    #         device.update_local_model(self.global_model)


def parse_command_line_args():
    parser = argparse.ArgumentParser(description="Server Configuration")

    # Server IP argument
    parser.add_argument(
        "-server-ip",
        dest="server_ip",
        type=str,
        help="The IP address this server should run on",
        required=True,
    )

    # Server Port argument
    parser.add_argument(
        "-server-port",
        dest="server_port",
        type=int,
        help="The port this server should run on",
        required=True,
    )

    # Verbose argument
    parser.add_argument(
        "-verbose",
        dest="verbose",
        action="store_true",
        help="Whether to run the server in verbose mode",
    )

    # Missingness mode argument
    parser.add_argument(
        "-missing",
        dest="missing",
        type=str,
        choices=["no-missing", "missing", "corrected"],
        help="The mode for handling missing data",
    )

    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parse_command_line_args()

    # # Access the parsed arguments
    server_ip = args.server_ip
    server_port = args.server_port
    verbose_mode = args.verbose

    server = Server(host=server_ip, port=server_port, missing_mode=args.missing)

    print("Starting Server", flush=True)

    server_thread = threading.Thread(target=server.serve)
    server_thread.start()

    # server.serve()

    from time import sleep

    sleep(200)

    # server.train(2, 0.5, 1)

    num_epochs = 1
    for i in range(num_epochs):
        print(f"Beginning epoch {i + 1} of {num_epochs}")
        server.survey_devices()
        server.train(2500, 0, 16)

    server.test(500)

    server.send_shutdown()

    shutdown = True

    server_thread.join()

    exit()
