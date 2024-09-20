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

        if failures:
            print(f"Round {server_round} had failures: {failures}")

        # Collect positions from all clients
        positions_list = [parameters[0] for _, parameters, _, _ in results]

        # Identify common positions
        positions_to_keep = self.identify_common_positions(positions_list)

        # Send back positions to keep to all clients
        return [positions_to_keep], {}

    def identify_common_positions(self, positions_list):
        num_clients = len(positions_list)
        tolerance = 1e-5  # Adjust as needed
        desired_num_voxels_per_axis = 50  # Adjust based on desired resolution

        # Compute dynamic voxel size
        all_positions = np.vstack(positions_list)
        min_coords = np.min(all_positions, axis=0)
        max_coords = np.max(all_positions, axis=0)
        bbox_size = max_coords - min_coords
        voxel_size = bbox_size / desired_num_voxels_per_axis
        voxel_size = np.mean(voxel_size)

        # Prepare data structures
        voxel_positions = defaultdict(lambda: defaultdict(list))  # {voxel_idx: {client_id: [positions]}}

        # Populate voxel_positions
        for client_id, positions in enumerate(positions_list):
            for pos in positions:
                voxel_idx = tuple(np.floor_divide(pos, voxel_size).astype(int))
                voxel_positions[voxel_idx][client_id].append(pos)

        # Identify positions to keep for each client
        client_positions_to_keep = {client_id: [] for client_id in range(len(positions_list))}

        for voxel_idx, client_pos_dict in voxel_positions.items():
            # Only consider voxels where all clients have positions
            if len(client_pos_dict) == num_clients:
                # For each client, collect positions in this voxel
                positions_per_client = {cid: np.array(pos_list) for cid, pos_list in client_pos_dict.items()}
                # For each client, identify positions that have matches in other clients
                for client_id in positions_list.keys():
                    own_positions = positions_per_client[client_id]
                    other_clients = [cid for cid in range(len(positions_list)) if cid != client_id]
                    # For each of own positions, check if there's a matching position in other clients
                    for pos in own_positions:
                        match = True
                        for other_client_id in other_clients:
                            other_positions = positions_per_client[other_client_id]
                            distances = np.linalg.norm(other_positions - pos, axis=1)
                            if not np.any(distances <= tolerance):
                                match = False
                                break
                        if match:
                            client_positions_to_keep[client_id].append(pos)

            # Convert lists to numpy arrays
        for client_id in client_positions_to_keep:
            client_positions_to_keep[client_id] = np.array(client_positions_to_keep[client_id])

        return client_positions_to_keep


def fit_config(server_round: int) -> Dict[str, Scalar]:
    config = {"round": server_round}
    return config

def on_evaluate_config(server_round: int) -> Dict[str, Scalar]:
    return {"round": server_round}

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
            on_evaluate_config_fn=on_evaluate_config,
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
