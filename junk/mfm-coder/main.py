from math import sqrt
from sys import argv
from typing import Tuple, Any

import cv2
import torch
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F

BOX_SIZE = 128
DISTANCE_PER_PIXEL = 0.0000002
SCAN_TARGET_SIZE = 64 * BOX_SIZE
GRID_SIZE = SCAN_TARGET_SIZE // BOX_SIZE
ARROW_MAGNITUDE = 0.5
ARROW_COLOR = (0, 0, 255)
ARROW_THICKNESS = 4
ENVIRONMENT_FACTOR = 0.5


def load_and_preprocess_image(image_path: str) -> Tuple[torch.Tensor, Any]:
    """Load image and prepare both display and analysis versions."""
    image = torchvision.io.image.read_image(image_path)
    image = TF.resize(image, [SCAN_TARGET_SIZE, SCAN_TARGET_SIZE])

    display_image = TF.rgb_to_grayscale(image[:3, :, :], num_output_channels=3)
    display_image = display_image.permute(1, 2, 0).to(torch.float).numpy() / 255.0

    grayscale_image = TF.rgb_to_grayscale(image[:3, :, :], num_output_channels=1)

    return grayscale_image, display_image


def split_image_into_patches(image: torch.Tensor) -> torch.Tensor:
    """Reshape image into a grid of patches."""
    return image.view(
        1,
        GRID_SIZE,
        BOX_SIZE,
        GRID_SIZE,
        BOX_SIZE
    ).permute(0, 1, 3, 2, 4)


def create_sobel_kernels() -> Tuple[torch.Tensor, torch.Tensor]:
    """Create Sobel kernels for gradient computation."""
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32
    ).view(1, 1, 3, 3)

    return sobel_x, sobel_y


def compute_gradient(patch: torch.Tensor, sobel_x: torch.Tensor, sobel_y: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor]:
    """Compute gradient direction for a patch using Sobel operators."""
    grad_x = F.conv2d(patch, sobel_x, padding=1)
    grad_y = F.conv2d(patch, sobel_y, padding=1)

    gradient = torch.stack([grad_x.mean(), grad_y.mean()])
    return gradient / gradient.norm(), gradient.norm()


def calculate_arrow_endpoints(direction: torch.Tensor, magnitude: torch.Tensor, col: int, row: int) -> Tuple[
    Tuple[int, int], Tuple[int, int]]:
    """Calculate start and end points for gradient arrow."""
    angle_rad = torch.atan2(direction[1], direction[0])
    # magnitude = 1
    patch_center = BOX_SIZE // 2
    arrow_length = (BOX_SIZE // 2) * ARROW_MAGNITUDE
    if magnitude < 0.5:
        arrow_length = 0
    magnitude = 1
    local_center = (patch_center, patch_center)
    local_start = (
        int(local_center[0] - arrow_length * torch.cos(angle_rad) * magnitude),
        int(local_center[1] - arrow_length * torch.sin(angle_rad) * magnitude)
    )

    local_end = (
        int(local_center[0] + arrow_length * torch.cos(angle_rad) * magnitude),
        int(local_center[1] + arrow_length * torch.sin(angle_rad) * magnitude)
    )

    global_start = (local_start[0] + col * BOX_SIZE, local_start[1] + row * BOX_SIZE)
    global_end = (local_end[0] + col * BOX_SIZE, local_end[1] + row * BOX_SIZE)

    return global_start, global_end


def visualize_gradients(image: torch.Tensor, display_image: Any) -> Any:
    """Process image patches and draw gradient arrows on display image."""
    patches = split_image_into_patches(image)
    sobel_x, sobel_y = create_sobel_kernels()

    dm = torch.tensor([[[0, 0, 0]] * GRID_SIZE] * GRID_SIZE, dtype=torch.float)

    for row in range(GRID_SIZE - 1):
        for col in range(GRID_SIZE - 1):
            patch = patches[:, row, col].reshape(BOX_SIZE, BOX_SIZE).to(torch.float)
            patch = patch.unsqueeze(0).unsqueeze(0)

            direction, magnitude = compute_gradient(patch, sobel_x, sobel_y)
            dm[row, col] = torch.tensor([direction[0], direction[1], magnitude])

    dm2 = dm.clone()

    # for row in range(GRID_SIZE - 1):
    #     for col in range(GRID_SIZE - 1):
    #         row_indices = torch.arange(GRID_SIZE).unsqueeze(1).expand(-1, GRID_SIZE)
    #         col_indices = torch.arange(GRID_SIZE).unsqueeze(0).expand(GRID_SIZE, -1)
    #         row_filter = (row_indices != row)
    #         col_filter = (col_indices != col)
    #         cf = row_filter & col_filter
    #         row_indices = row_indices[row_indices != row].to(torch.float64)
    #         col_indices = col_indices[col_indices != col].to(torch.float64)
    #         distances = torch.sqrt((row - row_indices) ** 2 + (col - col_indices) ** 2)
    #         scale_factor = 1 / ((BOX_SIZE * DISTANCE_PER_PIXEL * distances) ** 3)
    #         neighbors = dm.clone()
    #         neighbors[cf, 2] = 0
    #         neighbors[:, :, 2] *= scale_factor
    #         neighbors = neighbors.reshape(-1, 3)
    #         # neighbors = torch.stack(neighbors)  # N x 3
    #         own_rad = torch.atan2(dm[row, col][1], dm[row, col][0])
    #         neighbor_rad = torch.atan2(neighbors[:, 1], neighbors[:, 0])
    #         neighbor_rad = torch.sum(neighbor_rad * neighbors[:, 2]) / torch.sum(neighbors[:, 2])
    #         total_rad = (own_rad * (1 - ENVIRONMENT_FACTOR) + neighbor_rad * ENVIRONMENT_FACTOR)
    #         dm2[row, col, :2] = torch.tensor([torch.cos(total_rad), torch.sin(total_rad)])
    #         # dm2[row, col, 2:] = neighbors[:, 2:].mean(dim=0)

    for row in range(GRID_SIZE - 1):
        for col in range(GRID_SIZE - 1):
            x, y, m = dm2[row, col].tolist()
            direction = torch.tensor([x, y])
            magnitude = torch.tensor(m)
            start_point, end_point = calculate_arrow_endpoints(direction, magnitude, col, row)
            cv2.arrowedLine(display_image, start_point, end_point, ARROW_COLOR, ARROW_THICKNESS)

    return display_image


def display_and_wait(image: Any) -> None:
    """Display image and wait for user to quit."""
    cv2.imshow("Image", image)
    if cv2.waitKey(0) == ord("q"):
        exit(0)


def main():
    """Main function to process and visualize image gradients."""
    grayscale_image, display_image = load_and_preprocess_image(argv[1])
    display_image = visualize_gradients(grayscale_image, display_image)
    display_and_wait(display_image)


if __name__ == "__main__":
    main()
