import torch
import numpy as np
import torch.nn as nn

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Initialize SAM model
sam_checkpoint = "path/to/sam_vit_h_4b8939.pth"  # Replace with your SAM checkpoint
model_type = "vit_h"  # Options: "vit_t", "vit_b", "vit_l", "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to("cuda")  # Move to GPU if available

# Initialize mask generator
mask_generator = SamAutomaticMaskGenerator(sam)


def generate_segmentation_masks(image):
    """
    Generates segmentation masks for a given image using SAM.

    Args:
        image (numpy.ndarray): The input image in RGB format.

    Returns:
        List[Dict]: A list of mask dictionaries containing segmentation data.
    """
    masks = mask_generator.generate(image)
    return masks

def project_splats_to_image(splats, camtoworld, K):
    """
    Projects 3D splat positions to 2D image coordinates.

    Args:
        splats (torch.Tensor): [N, 3] tensor of splat positions in world coordinates.
        camtoworld (torch.Tensor): [4, 4] camera-to-world matrix.
        K (torch.Tensor): [3, 3] camera intrinsic matrix.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (u, v) pixel coordinates of splats.
    """
    # Convert to homogeneous coordinates
    ones = torch.ones((splats.shape[0], 1), device=splats.device)
    splats_h = torch.cat([splats, ones], dim=1).T  # [4, N]

    # Compute world-to-camera matrix
    world_to_cam = torch.inverse(camtoworld)[:3, :]  # [3, 4]

    # Project points
    pixels = K @ world_to_cam @ splats_h  # [3, N]
    pixels /= pixels[2, :]  # Normalize

    u, v = pixels[0, :], pixels[1, :]
    return u, v

def assign_segmentation_labels_to_splats(runner, image_idx, masks):
    """
    Assigns segmentation labels to splats based on image segmentation masks.

    Args:
        runner (Runner): Your existing Runner instance.
        image_idx (int): Index of the current image/view.
        masks (List[Dict]): List of segmentation masks for the image.
    """
    # Load camera parameters for the current image
    camtoworld = runner.trainset.camtoworlds[image_idx]  # [4, 4]
    K = runner.trainset.Ks[image_idx]  # [3, 3]

    # Get splat positions
    splats = runner.splats["means3d"]  # [N, 3]

    # Project splats to image plane
    u, v = project_splats_to_image(splats, camtoworld, K)  # [N], [N]

    # Initialize segmentation labels
    segmentation_labels = -1 * torch.ones(splats.shape[0], dtype=torch.int32, device=splats.device)

    for mask in masks:
        # Get mask properties
        mask_pixels = mask['segmentation']  # Binary mask [H, W]
        bbox = mask['bbox']  # [x, y, width, height]
        label = mask.get('label', -1)  # Use a default or predefined label

        x_min, y_min, width, height = bbox
        x_max, y_max = x_min + width, y_min + height

        # Find splats projecting within the mask bbox
        within_bbox = (u >= x_min) & (u <= x_max) & (v >= y_min) & (v <= y_max)
        splat_indices = torch.nonzero(within_bbox).squeeze()

        if splat_indices.numel() == 0:
            continue

        # Further check if the splat pixel is within the mask
        splat_u = u[splat_indices].long()
        splat_v = v[splat_indices].long()

        # Ensure indices are within image bounds
        splat_u = splat_u.clamp(0, mask_pixels.shape[1] - 1)
        splat_v = splat_v.clamp(0, mask_pixels.shape[0] - 1)

        in_mask = mask_pixels[splat_v, splat_u].bool()
        final_indices = splat_indices[in_mask]

        # Assign label
        segmentation_labels[final_indices] = label

    # Update splat descriptors with segmentation labels
    runner.splats["segmentation"] = torch.nn.Parameter(segmentation_labels)


def aggregate_segmentation_labels(runner, all_image_masks):
    """
    Aggregates segmentation labels from all views for each splat.

    Args:
        runner (Runner): Your existing Runner instance.
        all_image_masks (Dict[int, List[Dict]]): Mapping from image index to list of masks.

    """
    num_splats = runner.splats["means3d"].shape[0]
    label_counts = torch.zeros((num_splats, runner.num_classes), dtype=torch.int32, device=runner.device)

    for image_idx, masks in all_image_masks.items():
        assign_segmentation_labels_to_splats(runner, image_idx, masks)
        current_labels = runner.splats["segmentation"].cpu()
        for label in range(runner.num_classes):
            label_counts[:, label] += (current_labels == label).int()

    # Assign the label with the highest count
    final_labels = torch.argmax(label_counts, dim=1)
    runner.splats["segmentation"] = torch.nn.Parameter(final_labels)




def compute_segmentation_loss(runner, target_labels):
    """
    Computes the segmentation loss.

    Args:
        runner (Runner): Your existing Runner instance.
        target_labels (torch.Tensor): [N] tensor of target segmentation labels.

    Returns:
        torch.Tensor: Segmentation loss.
    """
    # Initialize the loss function
    segmentation_loss_fn = nn.CrossEntropyLoss()

    predicted_labels = runner.splats["segmentation"]  # [N]
    loss = segmentation_loss_fn(predicted_labels, target_labels)
    return loss