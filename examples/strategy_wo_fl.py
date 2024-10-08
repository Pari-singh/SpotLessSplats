import os
import numpy as np
import torch
from collections import defaultdict

num_clients = 3
num_voxels_per_axis = 2000
round = 1
tolerance = 1e-7
ckpt_c0 = f"outputs/lr_str2_20vox_manual/client_0/{round}/ckpts/ckpt_29999.pt"
ckpt_c1 = f"outputs/lr_str2_20vox_manual/client_1/{round}/ckpts/ckpt_29999.pt"
ckpt_c2 = f"outputs/lr_str2_20vox_manual/client_2/{round}/ckpts/ckpt_29999.pt"
client_pos = {0: ckpt_c0, 1: ckpt_c1, 2: ckpt_c2}

def extract_mean_positions(pt_file):
    data = torch.load(pt_file)
    splats = data['splats']

    # Extract data
    means3d = splats['means3d'].cpu().numpy()  # Positions (N, 3)
    print(len(means3d))
    return means3d

def get_positions_mask_using_kdtree(current_positions, positions_to_keep, tolerance):
    from scipy.spatial import cKDTree
    # Convert to numpy arrays
    current_positions_np = current_positions.cpu().detach().numpy()
    positions_to_keep_np = positions_to_keep.cpu().detach().numpy()

    # Build KD-tree
    tree = cKDTree(positions_to_keep_np)
    # Query tree for all current positions
    distances, indices = tree.query(current_positions_np, distance_upper_bound=tolerance)
    # Create mask
    mask = torch.tensor(distances <= tolerance)
    return mask

def update_model_per_client(client_id, positions):
    data = torch.load(client_pos[client_id])
    splats = data['splats']
    current_positions = splats['means3d']
    positions_to_keep = torch.tensor(positions)
    mask = get_positions_mask_using_kdtree(current_positions, positions_to_keep, tolerance=tolerance)
    # Filter splats based on mask
    for key in splats:
        splats[key] = splats[key][mask]
    print("Positions sent ", len(positions))
    print("Positions kept for next round ", len(splats['means3d']))
    save_splats_as_pth(splats, client_id)

#TODO: Update it
def save_splats_as_pth(splats, client_id):
    # Define the file path
    filename = os.path.join(f"outputs/lr_str2_{num_voxels_per_axis}vox_manual_tol-7/client_{client_id}/server/{round}/ckpt_29999.pt")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Save the splats
    torch.save({"step": 29999,
                "splats": splats}, filename)
    print(f"Saved splats to {filename}")

def process_per_client_positions(per_client_positions, num_voxels):
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
    print("Bbox size is: ", bbox_size)

    # Handle case where bbox_size is zero (all points are at the same coordinate)
    voxel_size_per_axis = bbox_size / desired_num_voxels_per_axis
    voxel_size = np.mean(voxel_size_per_axis)
    if voxel_size == 0:
        voxel_size = 1e-6  # Small number to prevent division by zero
    print(f"Voxel size: {voxel_size}")
    print(f"total # voxels: ")

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

def aggregate_fit():
    per_client_positions= {}
    for client_id, ckpt_path in enumerate([ckpt_c0, ckpt_c1, ckpt_c2]):
        per_client_positions[client_id] = extract_mean_positions(ckpt_path)

    per_client_processed_positions = process_per_client_positions(per_client_positions, num_voxels_per_axis)
    for client_id, positions in per_client_processed_positions.items():
        update_model_per_client(client_id, positions)

    # visualize_common_voxels(per_client_positions, common_voxels, voxel_dict, min_coords, voxel_size)



aggregate_fit()