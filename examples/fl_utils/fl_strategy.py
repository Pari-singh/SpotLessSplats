import os
from collections import defaultdict
import flwr
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
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
import open3d as o3d
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
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            print(f"Round {server_round} had failures: {failures}")

        # Extract positions lists from clients
        positions_list = [
            parameters_to_ndarrays(fit_res.parameters)[0]  # Assuming each client sends one array
            for client_proxy, fit_res in results
        ]

        if not positions_list:
            print("positions_list is empty.")
            return None, {}

        # Identify common positions
        per_client_positions = {}  # {client_id: positions}

        for client_proxy, fit_res in results:
            client_id = client_proxy.cid  # Assuming cid is the client ID
            client_positions = parameters_to_ndarrays(fit_res.parameters)[0]
            per_client_positions[client_id] = client_positions

        per_client_processed_positions = self.process_per_client_positions(per_client_positions)

        # Save per-client positions to files
        self.save_per_client_positions(per_client_processed_positions, server_round)
        return None, {}

    def process_per_client_positions(self, per_client_positions: Dict[str, np.ndarray], num_voxels: int) -> Dict[str, np.ndarray]:
        """
            Processes per-client positions to identify common voxels where all clients have positions.

            Args:
                per_client_positions (Dict[str, np.ndarray]): Dictionary mapping client IDs to their positions.
                num_voxels (int) : Desired_num_voxels
            Returns:
                Dict[str, np.ndarray]: Dictionary mapping client IDs to their processed positions in common voxels.
            """
        num_clients = len(per_client_positions)
        positions_list = list(per_client_positions.values())
        client_ids = list(per_client_positions.keys())

        # Stack all positions vertically to compute overall min and max coordinates
        all_positions = np.vstack(positions_list)  # Shape: (total_num_positions, 3)
        print(f"All positions shape: {all_positions.shape}")

        desired_num_voxels_per_axis = num_voxels
        min_coords = np.min(all_positions, axis=0)
        max_coords = np.max(all_positions, axis=0)
        bbox_size = max_coords - min_coords

        # Handle case where bbox_size is zero (all points are at the same coordinate)
        voxel_size_per_axis = bbox_size / desired_num_voxels_per_axis
        voxel_size = np.mean(voxel_size_per_axis)
        if voxel_size == 0:
            voxel_size = 1e-6  # Small number to prevent division by zero
        print(f"Voxel size: {voxel_size}")

        # Create a mapping from voxel index to clients present and positions per client
        voxel_dict = {}  # {voxel_idx: {'clients': set of client_ids, 'positions': {client_id: [positions]}}}

        for client_id, positions in per_client_positions.items():
            # Shift positions by min_coords to ensure all positions are positive
            shifted_positions = positions - min_coords
            # Compute voxel indices
            voxel_indices = np.floor_divide(shifted_positions, voxel_size).astype(int)
            for idx, voxel_idx in enumerate(map(tuple, voxel_indices)):
                if voxel_idx not in voxel_dict:
                    voxel_dict[voxel_idx] = {'clients': set(), 'positions': defaultdict(list)}
                voxel_dict[voxel_idx]['clients'].add(client_id)
                # Store the original position (not shifted)
                voxel_dict[voxel_idx]['positions'][client_id].append(positions[idx])

        # Identify voxels where all clients have positions
        common_voxels = [voxel_idx for voxel_idx, data in voxel_dict.items() if len(data['clients']) == num_clients]
        print(f"Number of common voxels: {len(common_voxels)}")

        # For each client, collect their positions in the common voxels
        per_client_processed_positions = {client_id: [] for client_id in client_ids}

        for voxel_idx in common_voxels:
            data = voxel_dict[voxel_idx]
            # For each client, collect their positions in this voxel
            for client_id in client_ids:
                positions_in_voxel = data['positions'][client_id]
                per_client_processed_positions[client_id].extend(positions_in_voxel)

        # Convert lists to numpy arrays
        for client_id in per_client_processed_positions:
            positions = per_client_processed_positions[client_id]
            per_client_processed_positions[client_id] = np.array(positions) if positions else np.empty((0, 3))
            print(f"Client {client_id} has {len(positions)} positions in common voxels.")

        return per_client_processed_positions


    def save_per_client_positions(self, per_client_positions: Dict[str, np.ndarray], server_round: int):
        import os
        output_dir = self.cfg.result_dir  # Assuming cfg has result_dir attribute
        os.makedirs(output_dir, exist_ok=True)

        for client_id, positions in per_client_positions.items():
            filename = os.path.join(output_dir, f"client_{client_id}_positions_round_{server_round}.npy")
            np.save(filename, positions)
            print(f"Saved positions for client {client_id} to {filename}")


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
            cfg=cfg,
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
            initial_parameters=flwr.common.ndarrays_to_parameters(server_model_static_params),
            cfg=cfg
        )

    return strategy
