import os

import flwr
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    Metrics,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import numpy as np
from typing import Callable, Union, Dict, List, Optional, Tuple
from collections import OrderedDict
import torch

from examples.utils import SpotLessModule


def get_evaluate_fn():
    """This is a function that returns a function. The returned
    function (i.e. `evaluate_fn`) will be executed by the strategy
    at the end of each round to evaluate the stat of the global
    model."""

    def evaluate_fn(server_round: int, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        Then, the model will be evaluated on the test set (recall we don't need
        this for Gaussian model training)."""

        return None

    return evaluate_fn

def weighted_average(metrics_list):

    # Initialize variables to store the sum of weighted metrics and total weights (samples)
    total_weight = 0
    aggregated_metrics = {}
    # Loop over each client's metrics and sample count
    for weight, metrics in metrics_list:
        total_weight += weight

        for metric_key, metric_value in metrics.items():
            if metric_key in aggregated_metrics:
                aggregated_metrics[metric_key] += metric_value * weight
            else:
                aggregated_metrics[metric_key] = metric_value * weight

    if total_weight > 0:
        for metric_key in aggregated_metrics:
            aggregated_metrics[metric_key] /= total_weight

    return aggregated_metrics
class SaveModelStrategy(flwr.server.strategy.FedAvg):
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = flwr.common.parameters_to_ndarrays(aggregated_parameters)
            mlp_model = None
            if self.mlp_spotless:
                mlp_model = SpotLessModule(
                    num_classes=1, num_features=self.mlp_spotless_num_feats
                ).cuda()

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(mlp_model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            mlp_model.load_state_dict(state_dict, strict=True)

            # Save the model
            server_path = os.path.join(self.cfg.result_dir, "server")
            os.makedirs(server_path, exist_ok=True)
            torch.save(mlp_model.state_dict(),
                       f"{server_path}/round_{server_round + (self.cfg.resume_round - 1)}.pth")

        return aggregated_parameters, aggregated_metrics

    def init_central_params(self, cfg):
        self.cfg = cfg
        self.mlp_spotless = cfg.semantics and not cfg.cluster
        self.mlp_spotless_num_feats = cfg.mlp_spotless_num_feats
        self.device = cfg.device

def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "round": server_round,
        "epochs": 1,  # Number of local epochs done by clients
        # "lr": 0.01,  # Learning rate to use by clients during fit()
    }
    return config

def choose_strategy(cfg, fit_config, weighted_average,
                    server_model_static_params=None):
    NUM_CLIENTS = cfg.num_clients
    if not (cfg.resume_round > 1):
        strategy = SaveModelStrategy(
            fraction_fit=cfg.fraction_fit,
            fraction_evaluate=cfg.fraction_evaluate,
            min_fit_clients=cfg.min_fit_clients,
            min_evaluate_clients=cfg.min_evaluate_clients,
            min_available_clients=int(
                NUM_CLIENTS * cfg.min_available_clients
            ),
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn=get_evaluate_fn(),
        )
    else:
        print("Initial Parameters given")
        strategy = SaveModelStrategy(
            fraction_fit=cfg.fraction_fit,
            fraction_evaluate=cfg.fraction_evaluate,
            min_fit_clients=cfg.min_fit_clients,
            min_evaluate_clients=cfg.min_evaluate_clients,
            min_available_clients=int(
                NUM_CLIENTS * cfg.min_available_clients
            ),
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn=get_evaluate_fn(),
            initial_parameters=flwr.common.ndarrays_to_parameters(server_model_static_params)
        )

    return strategy
