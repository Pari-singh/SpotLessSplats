import os
import pdb
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
        positions_to_keep, voxel_size = self.identify_common_positions(positions_list, self.cfg.num_voxels_per_axis)

        # Ensure positions_to_keep is a NumPy array
        if not isinstance(positions_to_keep, np.ndarray):
            print("positions_to_keep is not a NumPy array. Converting.")
            positions_to_keep = np.array(positions_to_keep)

        # Convert positions_to_keep to Parameters object
        aggregated_parameters = ndarrays_to_parameters([positions_to_keep])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions_to_keep)
        # Saving positions as ply
        server_path = os.path.join(self.cfg.result_dir, "server")
        os.makedirs(server_path, exist_ok=True)
        o3d.io.write_point_cloud(f"{server_path}/round_{server_round + (self.cfg.resume_round - 1)}.ply", pcd)
        # Prepare the config dictionary
        config = {
            "voxel_size": voxel_size  # Ensure voxel_size is a scalar or serialized appropriately
        }
        return aggregated_parameters, config

    def identify_common_positions(self, positions_list, num_voxels):
        num_clients = len(positions_list)
        tolerance = 1e-5  # Adjust as needed
        desired_num_voxels_per_axis = num_voxels #50  # Adjust based on desired resolution

        # Compute dynamic voxel size
        all_positions = np.vstack(positions_list)

        min_coords = np.min(all_positions, axis=0)
        max_coords = np.max(all_positions, axis=0)
        bbox_size = max_coords - min_coords
        voxel_size = bbox_size / desired_num_voxels_per_axis
        print("box_size and voxel_size ", bbox_size, voxel_size)
        voxel_size = np.mean(voxel_size)

        # Prepare data structures
        voxel_positions = defaultdict(set)  # {voxel_idx: set of client_ids}

        # Populate voxel_positions
        for client_id, positions in enumerate(positions_list):
            for pos in positions:
                voxel_idx = tuple(np.floor_divide(pos, voxel_size).astype(int))
                voxel_positions[voxel_idx].add(client_id)

        # Identify voxels where all clients have positions
        common_voxels = [voxel_idx for voxel_idx, clients in voxel_positions.items() if len(clients) == num_clients]

        # Collect positions in common voxels
        print("Collecting positions in common voxels")
        positions_to_keep = []
        for voxel_idx in common_voxels:
            # Collect positions from all clients in this voxel
            voxel_positions_list = []
            for client_id, positions in enumerate(positions_list):
                for pos in positions:
                    pos_voxel_idx = tuple(np.floor_divide(pos, voxel_size).astype(int))
                    if pos_voxel_idx == voxel_idx:
                        voxel_positions_list.append(pos)
            # Optionally, compute the mean position or pick one representative
            avg_position = np.mean(voxel_positions_list, axis=0)
            positions_to_keep.append(avg_position)

        positions_to_keep = np.array(positions_to_keep)
        print(f"positions_to_keep shape: {positions_to_keep.shape}")
        return positions_to_keep, voxel_size


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
