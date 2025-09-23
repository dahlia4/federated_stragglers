from typing import Union, Optional

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import pandas as pd
from .net import Net, get_parameters, set_parameters, train, test
from .shadow_recovery import ShadowRecovery
import random
from .knobs import MISSING, COMPUTE_WEIGHTS

class MnarStrategy(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.shadow_recovery = None
        self.survey_responses = {}
        self.participating_clients = []
        self.client_ids = []
        

    def __repr__(self) -> str:
        return "MnarStrategy"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        net = Net()
        ndarrays = get_parameters(net)
        return ndarrays_to_parameters(ndarrays)

    def compute_weights(self, participating_clients, client_ids):
        if not COMPUTE_WEIGHTS:
            return [1] * len(participating_clients)
        if self.shadow_recovery is None:
            #survey_list = [v for k,v in self.survey_responses.items()]
            #print(survey_list)
            test = pd.concat(self.survey_responses)
            self.shadow_recovery = ShadowRecovery(
                "D2",
                "S",
                "R",
                ["D1"],
                pd.concat(self.survey_responses),
            )
            self.shadow_recovery._findRoots()

        res = []

        for client,id in zip(participating_clients,client_ids):
            score = self.shadow_recovery._propensityScoresRY(self.survey_responses[id])
            res.append(float((1 / score).iloc[0]))

        return res
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        if server_round % 200 == 1:
            self.participating_clients.clear()
            self.client_ids = []
            curr_id = 0
            ins = GetPropertiesIns({})
            print(len(client_manager.all().values()))
            for client in client_manager.all().values():
                in_data = client.get_properties(ins,timeout=30,group_id=str(server_round)).properties
                processed_in_data = {k: [v] for k,v in in_data.items()}
                client_dict = pd.DataFrame(data=processed_in_data)
                self.survey_responses[curr_id] = client_dict[["R", "S", "D1", "D2"]]
                if MISSING:
                    if client_dict["R"][0] == 1:
                        self.participating_clients.append(client)
                        self.client_ids.append(curr_id)
                else:
                    self.participating_clients.append(client)
                    self.client_ids.append(curr_id)
                        
                curr_id += 1
            print(len(self.participating_clients))
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            #client_manager.num_available()
            len(self.participating_clients)
        )
        weights = self.compute_weights(self.participating_clients, self.client_ids)
        clients = random.choices(self.participating_clients, k=sample_size, weights=weights)

        # Create custom configs
        standard_config = {"lr": 0.001}
        
        fit_configurations = []
        for idx, client in enumerate(clients):
            fit_configurations.append((client, FitIns(parameters, standard_config)))
        return fit_configurations
    def get_propensity_scores(self,results,failures):
        #Temporary, unfinished implementation, currently does nothing
        return [1] * len(results)
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        #Returns a list of propensity scores, with one per result
        propensity_scores = self.get_propensity_scores(results,failures)
        
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]



        #weights all parameters by the number of examples (related to total number of examples)
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        if server_round % 10 != 0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}
        print(results[0][1].loss)
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        if server_round % 10 == 0 and self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
            num = ""
            #with open("number.txt") as readfile:
            #    for line in readfile:
            #        num = line.strip()
            #if not MISSING:
            #    with open(f"single_results/res_not_missing_ends_{num}.txt","a") as writefile:
            #        writefile.write(f"{server_round}: {metrics_aggregated}\n")
            #elif not COMPUTE_WEIGHTS:
            #    with open(f"single_results/res_not_computed_ends_{num}.txt","a") as writefile:
            #        writefile.write(f"{server_round}: {metrics_aggregated}\n")
            #else:
            #    with open(f"single_results/res_computed_ends_{num}.txt","a") as writefile:
            #        writefile.write(f"{server_round}: {metrics_aggregated}\n")
            if server_round % 1000 == 0:
                with open("test_results/1000_clients.txt","a") as writefile:
                    writefile.write(f"{server_round}: {metrics_aggregated}\n") 
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
