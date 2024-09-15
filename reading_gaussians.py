import collections

import torch
import open3d as o3d
import numpy as np
import os
from plyfile import PlyData, PlyElement
from collections import OrderedDict

def normalize_sh0_for_colors(sh0):
    # Map sh0 values from [-1, 1] to [0, 255]
    sh0_normalized = (sh0 + 1) / 2  # Map to [0, 1]
    sh0_scaled = sh0_normalized * 255
    sh0_clipped = np.clip(sh0_scaled, 0, 255)
    return sh0_clipped.astype(np.uint8)

def check_and_fix_nan_inf(array, name):
    if np.isnan(array).any() or np.isinf(array).any():
        print(f"Warning: NaN or infinity detected in {name}, replacing with zeros.")
        array = np.nan_to_num(array)
    return array

def convert_pt_to_ply(pt_file, ply_file):
    # Load the .pt file
    data = torch.load(pt_file)
    splats = data['splats']

    # Extract data
    means3d = splats['means3d'].cpu().numpy()            # Positions (N, 3)
    opacities = splats['opacities'].cpu()                # Opacities (N,)
    quats = splats['quats'].cpu().numpy()                # Rotations (N, 4)
    scales = splats['scales'].cpu().numpy()              # Scales (N, 3)
    sh0 = splats['sh0'].cpu().numpy()                    # sh0 features (N, 1, num_feats)
    shN = splats['shN'].cpu().numpy()                    # shN features (N, 15, num_feats)

    N = means3d.shape[0]

    # Set normals to zeros
    normals = np.zeros_like(means3d)                     # Normals (N, 3)

    # Flatten sh0 and shN features
    f_dc = sh0.reshape(N, -1)                            # Flatten sh0 to (N, num_feats_sh0)
    f_rest = shN.reshape(N, -1)                          # Flatten shN to (N, 15 * num_feats_shN)
    print("f_dc shape:", f_dc.shape)
    print("f_rest shape:", f_rest.shape)

    # Handle opacities (apply logit function)
    # Avoid zeros or ones to prevent infinities
    eps = 1e-6
    opacities = opacities.clamp(eps, 1 - eps)
    opacities = torch.logit(opacities).cpu().numpy().reshape(-1, 1)
    print("opacities shape after logit:", opacities.shape)

    # Concatenate all attributes
    attributes = np.concatenate(
        (
            means3d,        # x, y, z          (N, 3)
            normals,        # nx, ny, nz       (N, 3)
            f_dc,           # f_dc_0, ...      (N, num_features_sh0)
            f_rest,         # f_rest_0, ...    (N, 15 * num_feats_shN)
            opacities,      # opacity          (N, 1)
            scales,         # scale_0, scale_1, scale_2 (N, 3)
            quats           # rot_0, rot_1, rot_2, rot_3 (N, 4)
        ),
        axis=1
    )
    print("attributes shape:", attributes.shape)

    # Construct list of attributes
    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(quats.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # Create the dtype for structured array
    attribute_names = construct_list_of_attributes()
    dtype_full = [(attribute, 'f4') for attribute in attribute_names]

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(ply_file), exist_ok=True)

    # Create structured array for PLY
    elements = np.empty(N, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    # Write the PLY file
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_file)
    print(f"Converted {pt_file} to {ply_file} with all required attributes including shN.")

def convert_ply_to_pt(ply_file, pt_file):
    # Read the PLY file
    plydata = PlyData.read(ply_file)
    vertex_data = plydata['vertex'].data

    # Extract attributes
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    positions = np.vstack((x, y, z)).T  # Shape: (N, 3)

    # Features (sh0 and shN)
    # Extract 'f_dc_*' attributes (corresponds to 'sh0')
    f_dc_keys = [key for key in vertex_data.dtype.names if key.startswith('f_dc_')]
    f_dc_keys_sorted = sorted(f_dc_keys, key=lambda x: int(x.split('_')[-1]))
    sh0 = np.vstack([vertex_data[key] for key in f_dc_keys_sorted]).T  # Shape: (N, num_features_sh0)

    # Extract 'f_rest_*' attributes (corresponds to 'shN')
    f_rest_keys = [key for key in vertex_data.dtype.names if key.startswith('f_rest_')]
    f_rest_keys_sorted = sorted(f_rest_keys, key=lambda x: int(x.split('_')[-1]))
    f_rest = np.vstack([vertex_data[key] for key in f_rest_keys_sorted]).T  # Shape: (N, total_features_shN)

    # Reshape f_rest back to (N, 15, num_feats)
    N = positions.shape[0]
    total_features_shN = f_rest.shape[1]
    num_feats_shN = total_features_shN // 15
    shN = f_rest.reshape(N, 15, num_feats_shN)

    # Opacities
    opacities = vertex_data['opacity']
    # Apply sigmoid to invert logit transformation
    opacities = torch.sigmoid(torch.from_numpy(opacities).float())

    # Scales
    scale_keys = ['scale_0', 'scale_1', 'scale_2']
    scales = np.vstack([vertex_data[key] for key in scale_keys]).T  # Shape: (N, 3)
    scales = torch.from_numpy(scales).float()

    # Rotations (quaternions)
    rot_keys = ['rot_0', 'rot_1', 'rot_2', 'rot_3']
    quats = np.vstack([vertex_data[key] for key in rot_keys]).T  # Shape: (N, 4)

    # Move tensors to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Reconstruct splats dictionary
    splats = {
        'means3d': torch.from_numpy(positions).float().to(device),
        'opacities': opacities.to(device),
        'quats': torch.from_numpy(quats).float().to(device),
        'scales': scales.to(device),
        'sh0': torch.from_numpy(sh0).float().unsqueeze(1).to(device),  # Add the dimension back
        'shN': torch.from_numpy(shN).float().to(device),
    }

    # Include 'steps' key
    steps = 29999  # Adjust as needed

    # Save the splats dictionary along with 'steps'
    data = {
        'splats': collections.OrderedDict(splats),
        'step': steps
    }

    # Save to .pt file
    torch.save(data, pt_file)
    print(f"Converted {ply_file} back to {pt_file} with all features including shN.")


if __name__=="__main__":
    pt_file = '/media/fast_data/rice_spotless/outputs/sofa_people_central_ubp3/ckpts/ckpt_29999.pt'
    ply_file = '/media/fast_data/rice_spotless/outputs/sofa_people_central_ubp3/ckpts/ckpt_29999.ply'
    new_ply_file = "/media/fast_data/rice_spotless/outputs/sofa_people_central_ubp3/ckpts/ckpt_29999.ply"
    new_pt_file = "/media/fast_data/rice_spotless/outputs/sofa_people_central_ubp3/ckpts/nosupersplat_ckpt_29999.pt"
    # convert_pt_to_ply(pt_file, ply_file)
    convert_ply_to_pt(new_ply_file, new_pt_file)