from typing import Callable, Optional

import torch
from torch.nn import functional
import einops

from . import filters


IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


def imagenet_normalization(x: torch.Tensor) -> torch.Tensor:
    """Applies normalization to an image tensor.

    :param x: Tensor of shape [B, T, 3, H, W].

    :return: Tensor of shape [B, T, 3, H, W] with normalization.
    """
    mean: torch.Tensor = torch.as_tensor(IMAGENET_MEAN).type_as(x).to(x.device)
    std: torch.Tensor = torch.as_tensor(IMAGENET_STD).type_as(x).to(x.device)
    x = torch.permute(x, (0, 1, -2, -1, -3))
    x = (x - mean) / std
    x = torch.permute(x, (0, 1, -1, -3, -2))
    return x


def remove_imagenet_normalization(x: torch.Tensor) -> torch.Tensor:
    """Removes the normalization from an image tensor.

    :param x: Tensor of shape [B, T, 3, H, W].

    :return: Tensor of shape [B, T, 3, H, W] without normalization.
    """
    mean: torch.Tensor = torch.as_tensor(IMAGENET_MEAN).type_as(x).to(x.device)
    std: torch.Tensor = torch.as_tensor(IMAGENET_STD).type_as(x).to(x.device)
    x = torch.permute(x, (0, 1, -2, -1, -3))
    x = x * std + mean
    x = torch.permute(x, (0, 1, -1, -3, -2))
    return x


def one_minus_one_normalization(x: torch.Tensor) -> torch.Tensor:
    """Applies normalization to an image tensor.

    :param x: Tensor of shape [B, T, 3, H, W].

    :return: Tensor of shape [B, T, 3, H, W] with normalization.
    """
    return x * 2.0 - 1.0


def remove_one_minus_one_normalization(x: torch.Tensor) -> torch.Tensor:
    """Removes the normalization from an image tensor.

    :param x: Tensor of shape [B, T, 3, H, W].

    :return: Tensor of shape [B, T, 3, H, W] without normalization.
    """
    return (x + 1.0) / 2.0


def spectral_mask_sequence(
    x: torch.Tensor,
    masking_radius: int = 16,
    clamp_min: float | None = 0.0,
    clamp_max: float | None = 1.0
) -> dict[str, torch.Tensor]:
    """Masks the low and high-frequencies of a spatiotemporal sequence."""
    height: int = x.size(-2)
    width: int = x.size(-1)
    max_size: int = max(height, width)
    batch_size: int = x.size(0)
    x = einops.rearrange(x, "b t c h w -> (b t) c h w")

    if height != width:
        x = functional.interpolate(x, size=(max_size, max_size), mode="bilinear")

    mask: torch.Tensor = filters.generate_circular_mask(max_size, masking_radius, device=x.device)
    x_low, x_high = filters.filter_image_frequencies(x, mask)
    if clamp_min is not None or clamp_max is not None:
        x_low = torch.clamp(x_low, min=clamp_min, max=clamp_max).to(x.dtype)
        x_high = torch.clamp(x_high, min=clamp_min, max=clamp_max).to(x.dtype)

    if height != width:
        x_low = functional.interpolate(x_low, size=(height, width), mode="bilinear")
        x_high = functional.interpolate(x_high, size=(height, width), mode="bilinear")

    x_low = einops.rearrange(x_low, "(b t) c h w -> b t c h w", b=batch_size)
    x_high = einops.rearrange(x_high, "(b t) c h w -> b t c h w", b=batch_size)

    masked_sequences: dict[str, torch.Tensor] = {
        f"spectral_low_{masking_radius}": x_low,
        f"spectral_high_{masking_radius}": x_high
    }

    return masked_sequences


def mask_sequence(
    x: torch.Tensor,
    masking_strategy: str,
    masking_length: int = 4,
    denorm: Optional[Callable[[torch.Tensor], torch.Tensor]] = remove_imagenet_normalization,
    norm: Optional[Callable[[torch.Tensor], torch.Tensor]] = imagenet_normalization,
    spectral_radius: int = 16
) -> dict[str, torch.Tensor]:
    if denorm is not None:
        x = denorm(x)

    masked_part: torch.Tensor = x[:, :masking_length]
    non_masked_part: torch.Tensor = x[:, masking_length:]

    if masking_strategy == "spectral":
        masked_sequences: dict[str, torch.Tensor] = spectral_mask_sequence(
            masked_part, spectral_radius
        )
    else:
        raise NotImplementedError(f"Masking strategy {masking_strategy} is not implemented.")

    masked_sequences = {name: torch.cat((seq, non_masked_part), dim=1)
                        for name, seq in masked_sequences.items()}

    if norm is not None:
        masked_sequences = {name: norm(seq) for name, seq in masked_sequences.items()}

    return masked_sequences


def display_image(x, batch_idx: int = 0, time_idx: int = 0) -> None:
    import numpy as np
    from PIL import Image
    img = x[batch_idx, time_idx].permute((1, 2, 0)).detach().cpu().numpy()
    img = (img * 225).astype(np.uint8)
    Image.fromarray(img).show()
