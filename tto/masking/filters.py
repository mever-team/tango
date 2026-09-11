from typing import Optional

import torch
from torch import fft
from torch import linalg


__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"
__version__: str = "1.0.0"
__revision__: int = 1


def filter_image_frequencies(
    image: torch.Tensor,
    mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filters the frequencies of an image according to the provided mask.

    Dimensionalities:
        - B is the batch size.
        - C is the number of image channels.
        - H is the height of the image.
        - W is the width of the image.

    :param image: The images to filter. Dimensionality: [[B] x C] x H x W
    :param mask: The mask that indicates the frequencies in the 2D DFT spectrum to be
        filtered out. Values of 1 in the mask indicate that the corresponding frequency will
        be allowed, while values of 0 indicate that the corresponding frequency will be
        filtered out. The mask is multiplied with the center-shifted spectrum of the image.
        Thus, it should follow this format. Dimensionality: [[B] x C] x H x W.

    :return: A tuple that includes:
        - The filtered image. Dimensionality: [[B] x C] x H x W
        - The residual of the filtered image (filtered with the inverse mask).
            Dimensionality: [[B] x C] x H x W
    """
    # Compute FFT.
    image = fft.fft2(image)
    image = fft.fftshift(image)

    # Filter image.
    filtered_image: torch.Tensor = image * mask
    residual_filtered_image: torch.Tensor = image * (1 - mask)

    # Compute IFFT.
    filtered_image = fft.ifftshift(filtered_image)
    filtered_image = fft.ifft2(filtered_image).real
    residual_filtered_image = fft.ifftshift(residual_filtered_image)
    residual_filtered_image = fft.ifft2(residual_filtered_image).real

    return filtered_image, residual_filtered_image


def generate_circular_mask(
    input_size: int,
    mask_radius_start: int,
    mask_radius_stop: Optional[int] = None,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    coordinates: torch.Tensor = generate_centered_2d_coordinates_grid(input_size, device)
    radius: torch.Tensor = linalg.vector_norm(coordinates, dim=-1)
    mask: torch.Tensor = torch.where(radius < mask_radius_start, 1, 0)
    if mask_radius_stop is not None:
        mask = torch.where(radius > mask_radius_stop, 1, mask)
    return mask


def generate_centered_2d_coordinates_grid(
    size: int,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    assert size % 2 == 0, "Input size must be even."
    coords_values: torch.Tensor = torch.arange(0, size // 2, dtype=torch.float, device=device)
    coords_values = torch.cat([coords_values.flip(dims=(0,)), coords_values], dim=0)
    coordinates_x: torch.Tensor = coords_values.unsqueeze(dim=0).expand(size, -1)
    coordinates_y: torch.Tensor = torch.t(coordinates_x)
    coordinates: torch.Tensor = torch.stack([coordinates_x, coordinates_y], dim=2)
    return coordinates
