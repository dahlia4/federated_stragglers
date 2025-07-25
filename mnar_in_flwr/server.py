from strategy import MnarStrategy
from flwr.common import Context
from flwr.server import ServerConfig, ServerApp, ServerAppComponents
from knobs import NUM_ROUNDS


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context):
    strategy = MnarStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn = weighted_average
    )
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=strategy, config=config)

server = ServerApp(server_fn=server_fn)
