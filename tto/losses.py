import copy
import dataclasses
import pathlib
import shutil
from typing import Callable, Any, Optional

import filelock
import einops
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from tto.masking import masking_strategies
from utils import csv_tools


def compute_tto_loss(
    spectral_loss_type: str,
    denoised_predictions: dict[str, torch.Tensor],
    masking_radius: int,
    original_spatial_similarity: torch.Tensor,
    original_initial_latent: torch.Tensor,
    decoder: Callable,
    encoder: Callable,
    verbose: bool = True
) -> torch.Tensor:
    if spectral_loss_type == "predicted_low_pass_predicted_high_pass":
        predicted_low_pass: torch.Tensor = denoised_predictions[f"spectral_low_{masking_radius}"]
        predicted_high_pass: torch.Tensor = denoised_predictions[f"spectral_high_{masking_radius}"]
        similarity: torch.Tensor = nn.functional.cosine_similarity(predicted_low_pass, predicted_high_pass, dim=-1)
        similarity_mean: torch.Tensor = similarity.mean((-1, -2, -3))
        if verbose and dist.is_initialized():
            print(f"Predicted Low Pass - Predicted High Pass (Rank {dist.get_node_local_rank()}): "
                  f"{similarity_mean.detach().cpu().item()}")
        elif verbose:
            print(f"Predicted Low Pass - Predicted High Pass: {similarity_mean.detach().cpu().item()}")
        spectral_loss: torch.Tensor = (1 - similarity_mean) ** 2
    elif spectral_loss_type == "predicted_high_pass_high_pass":
        predicted_high_pass: torch.Tensor = denoised_predictions[f"spectral_high_{masking_radius}"]

        # Compute low and high-pass filtered variants from the original.
        predicted_original: torch.Tensor = denoised_predictions[f"original"]
        decoded_original: torch.Tensor = checkpoint(decoder, predicted_original, use_reentrant=False)
        decoded_original = (decoded_original * 0.5 + 0.5).clamp(0, 1)
        # Spatially apply low and high-pass filtering to each frame and channel of the video.
        masked_batches: list[dict[str, torch.Tensor]] = []
        # Apply separately over each channel C to reduce required memory.
        for i in range(decoded_original.size(1)):
            # Sample is B x C x T x H x W, while spectral_mask_sequence expects B x T x C x H x W,
            # yet it works because internally it applies 2D FFT, i.e. only to the last two dimensions.
            masked_batch: dict[str, torch.Tensor] = masking_strategies.spectral_mask_sequence(
                decoded_original[:, i].unsqueeze(dim=1).to(torch.float32)
            )
            masked_batches.append(masked_batch)
        masked: dict[str, torch.Tensor] = {
            n: torch.cat([m[n] for m in masked_batches], dim=1).to(predicted_original.dtype)
            for n in masked_batches[0].keys()
        }
        masked = {n: checkpoint(encoder, t.permute([0, 2, 1, 3, 4]), use_reentrant=False)
                  for n, t in masked.items()}
        high_pass: torch.Tensor = masked[f"spectral_high_{masking_radius}"]

        similarity: torch.Tensor = nn.functional.cosine_similarity(predicted_high_pass, high_pass, dim=-1)
        similarity_mean: torch.Tensor = similarity.mean((-1, -2, -3))
        if verbose and dist.is_initialized():
            print(f"Predicted High Pass - High Pass (Rank {dist.get_node_local_rank()}): "
                  f"{similarity_mean.detach().cpu().item()}")
        elif verbose:
            print(f"Predicted High Pass - High Pass: {similarity_mean.detach().cpu().item()}")
        spectral_loss: torch.Tensor = (1 - similarity_mean) ** 2
    elif spectral_loss_type == "predicted_high_low_pass_high_low_pass":
        predicted_high_pass: torch.Tensor = denoised_predictions[f"spectral_high_{masking_radius}"]
        predicted_low_pass: torch.Tensor = denoised_predictions[f"spectral_low_{masking_radius}"]

        # Compute low and high-pass filtered variants from the original.
        predicted_original: torch.Tensor = denoised_predictions[f"original"]
        decoded_original: torch.Tensor = checkpoint(decoder, predicted_original, use_reentrant=False)
        # decoded_original: torch.Tensor = decoder(predicted_original)
        decoded_original = (decoded_original * 0.5 + 0.5).clamp(0, 1)
        # Spatially apply low and high-pass filtering to each frame and channel of the video.
        masked_batches: list[dict[str, torch.Tensor]] = []
        # Apply separately over each channel C to reduce required memory.
        for i in range(decoded_original.size(1)):
            # Sample is B x C x T x H x W, while spectral_mask_sequence expects B x T x C x H x W,
            # yet it works because internally it applies 2D FFT, i.e. only to the last two dimensions.
            masked_batch: dict[str, torch.Tensor] = masking_strategies.spectral_mask_sequence(
                decoded_original[:, i].unsqueeze(dim=1).to(torch.float32)
            )
            # masked_batch: dict[str, torch.Tensor] = checkpoint(masking_strategies.spectral_mask_sequence,
            #     decoded_original[:, i].unsqueeze(dim=1).to(torch.float32),
            #     use_reentrant=False
            # )
            masked_batches.append(masked_batch)
        masked: dict[str, torch.Tensor] = {
            n: torch.cat([m[n] for m in masked_batches], dim=1).to(predicted_original.dtype)
            for n in masked_batches[0].keys()
        }
        masked: dict[str, torch.Tensor] = {n: (video - 0.5) / 0.5 for n, video in masked.items()}  # Convert to [-1, 1].
        masked = {n: encoder(t.permute([0, 2, 1, 3, 4])) for n, t in masked.items()}
        # masked = {n: checkpoint(encoder, t.permute([0, 2, 1, 3, 4]), use_reentrant=False)
        #           for n, t in masked.items()}
        low_pass: torch.Tensor = masked[f"spectral_low_{masking_radius}"]
        high_pass: torch.Tensor = masked[f"spectral_high_{masking_radius}"]

        similarity_high: torch.Tensor = nn.functional.cosine_similarity(predicted_high_pass, high_pass, dim=-1)
        similarity_high_mean: torch.Tensor = similarity_high.mean((-1, -2, -3))
        similarity_low: torch.Tensor = nn.functional.cosine_similarity(predicted_low_pass, low_pass, dim=-1)
        similarity_low_mean: torch.Tensor = similarity_low.mean((-1, -2, -3))
        if verbose and dist.is_initialized():
            print(f"Predicted High Pass - High Pass (Rank {dist.get_node_local_rank()}): "
                  f"{similarity_high_mean.detach().cpu().item()}")
            print(f"Predicted Low Pass - Low Pass (Rank {dist.get_node_local_rank()}): "
                  f"{similarity_low_mean.detach().cpu().item()}")
        elif verbose:
            print(f"Predicted High Pass - High Pass: {similarity_high_mean.detach().cpu().item()}")
            print(f"Predicted Low Pass - Low Pass: {similarity_low_mean.detach().cpu().item()}")
        spectral_loss: torch.Tensor = (1 - similarity_high_mean) ** 2 + (1 - similarity_low_mean) ** 2
    else:
        raise NotImplementedError(f"{spectral_loss_type} is not a supported spectral loss.")

    # Retain at least the same cosine similarity as the unoptimized model.
    predicted_original: torch.Tensor = denoised_predictions[f"original"]
    current_spatial_similarity: torch.Tensor = torch.nn.functional.cosine_similarity(
        predicted_original, original_initial_latent.detach(), dim=-1
    ).mean((-1, -2, -3))
    spatial_d: torch.Tensor = squared_hinge_penalty(current_spatial_similarity , original_spatial_similarity)
    if verbose and dist.is_initialized():
        print(f"Original Spatial Similarity (Rank {dist.get_node_local_rank()}): "
              f"{original_spatial_similarity.detach().cpu().item()}")
        print(f"Current Spatial Similarity (Rank {dist.get_node_local_rank()}): "
              f"{current_spatial_similarity.detach().cpu().item()}")
        print(f"Spatial Distance (Rank {dist.get_node_local_rank()}): "
              f"{spatial_d.detach().cpu().item()}")
    elif verbose:
        print(f"Original Spatial Similarity: "
              f"{original_spatial_similarity.detach().cpu().item()}")
        print(f"Current Spatial Similarity: "
              f"{current_spatial_similarity.detach().cpu().item()}")
        print(f"Spatial Distance: "
              f"{spatial_d.detach().cpu().item()}")

    # TODO: Retain at least the same inter-frame cosine similarity as the condition frames.

    # condition_temporal_d =

    l: torch.Tensor = 3.0 * spatial_d + spectral_loss

    return l


def mask_latent_frames(
    latent_frames: torch.Tensor,
    decoder: Callable,
    encoder: Callable,
    use_gradient_checkpointing: bool = False
) -> dict[str, torch.Tensor]:
    if use_gradient_checkpointing:
        decoded_original: torch.Tensor = checkpoint(decoder, latent_frames, use_reentrant=False)
    else:
        decoded_original: torch.Tensor = decoder(latent_frames)
    decoded_original = (decoded_original * 0.5 + 0.5).clamp(0, 1)  # Convert to [0, 1].

    # Spatially apply low and high-pass filtering to each frame and channel of the video.
    masked_batches: list[dict[str, torch.Tensor]] = []
    # Apply separately over each channel C to reduce required memory.
    for i in range(decoded_original.size(1)):
        # Sample is B x C x T x H x W, while spectral_mask_sequence expects B x T x C x H x W,
        # yet it works because internally it applies 2D FFT, i.e. only to the last two dimensions.
        masked_batch: dict[str, torch.Tensor] = masking_strategies.spectral_mask_sequence(
            decoded_original[:, i].unsqueeze(dim=1).to(torch.float32)
        )
        # masked_batch: dict[str, torch.Tensor] = checkpoint(masking_strategies.spectral_mask_sequence,
        #     decoded_original[:, i].unsqueeze(dim=1).to(torch.float32),
        #     use_reentrant=False
        # )
        masked_batches.append(masked_batch)
    masked: dict[str, torch.Tensor] = {
        n: torch.cat([m[n] for m in masked_batches], dim=1).to(latent_frames.dtype)
        for n in masked_batches[0].keys()
    }
    masked: dict[str, torch.Tensor] = {n: (video - 0.5) / 0.5 for n, video in masked.items()}  # Convert to [-1, 1].
    masked = {n: encoder(t.permute([0, 2, 1, 3, 4])) for n, t in masked.items()}

    return masked


def compute_spatial_similarity(block1: torch.Tensor, block2: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(block1, block2, dim=-1).mean()


def squared_hinge_penalty(x, t, lam=1.0):
    diff = torch.clamp(t - x, min=0.0)
    return lam * (diff ** 2).mean()


def compute_spectral_masked_similarities(
    denoised_predictions: dict[str, torch.Tensor],
    masking_radius: int,
    decoder: Callable,
    encoder: Callable,
    tag: str
) -> dict[str, torch.Tensor]:
    predicted_original: torch.Tensor = denoised_predictions["original"]
    predicted_low_pass: torch.Tensor = denoised_predictions[f"spectral_low_{masking_radius}"]
    predicted_high_pass: torch.Tensor = denoised_predictions[f"spectral_high_{masking_radius}"]

    # Compute low and high-pass filtered variants from the original.
    decoded_original: torch.Tensor = decoder(predicted_original)
    decoded_original = (decoded_original * 0.5 + 0.5).clamp(0, 1)
    # Spatially apply low and high-pass filtering to each frame and channel of the video.
    masked_batches: list[dict[str, torch.Tensor]] = []
    # Apply separately over each channel C to reduce required memory.
    for i in range(decoded_original.size(1)):
        # Sample is B x C x T x H x W, while spectral_mask_sequence expects B x T x C x H x W,
        # yet it works because internally it applies 2D FFT, i.e. only to the last two dimensions.
        masked_batch: dict[str, torch.Tensor] = masking_strategies.spectral_mask_sequence(
            decoded_original[:, i].unsqueeze(dim=1).to(torch.float32)
        )
        masked_batches.append(masked_batch)
    masked: dict[str, torch.Tensor] = {
        n: torch.cat([m[n] for m in masked_batches], dim=1).to(predicted_original.dtype)
        for n in masked_batches[0].keys()
    }
    masked = {n: encoder(t.permute([0, 2, 1, 3, 4])) for n, t in masked.items()}
    low_pass: torch.Tensor = masked[f"spectral_low_{masking_radius}"]
    high_pass: torch.Tensor = masked[f"spectral_high_{masking_radius}"]

    # Compute similarities.
    similarities_to_compute: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("predicted_original_predicted_low_pass", predicted_original, predicted_low_pass),
        ("predicted_original_predicted_high_pass", predicted_original, predicted_high_pass),
        ("predicted_low_pass_predicted_high_pass", predicted_low_pass, predicted_high_pass),
        ("predicted_low_pass_low_pass", predicted_low_pass, low_pass),
        ("predicted_high_pass_high_pass", predicted_high_pass, high_pass),
    ]
    similarities: dict[str, torch.Tensor] = {}
    for similarity_label, t1, t2 in similarities_to_compute:
        similarity: torch.Tensor = nn.functional.cosine_similarity(t1, t2, dim=-1)
        similarity_mean: torch.Tensor = similarity.mean((-1, -2, -3))
        similarity_std: torch.Tensor = similarity.std((-1, -2, -3))

        similarities[f"{tag}_{similarity_label}_mean"] = similarity_mean
        similarities[f"{tag}_{similarity_label}_std"] = similarity_std

    return similarities


# def compute_spectral_masked_similarities(
#     pipeline: Video2WorldPipeline,
#     step: int,
#     sample: dict[str, torch.Tensor],
#     x0_fn: dict[str, Callable],
#     sigma_in: torch.Tensor,
#     output_path: pathlib.Path | None = None,
#     similarities_storage: 'SimilarityStorage | None' = None,
#     export_decoded_videos: bool = True
# ) -> None:
#     from torch import nn
#
#     mask_radius: int = 16
#
#     predicted_original: torch.Tensor = sample["original"]
#     predicted_low_pass: torch.Tensor = sample[f"spectral_low_{mask_radius}"]
#     predicted_high_pass: torch.Tensor = sample[f"spectral_high_{mask_radius}"]
#
#     # Compute low and high-pass filtered variants from the original (in the latent space).
#     # TODO: Check if it is better to perform masking in the RGB space, i.e. decode to RGB, mask
#     #  and then re-encode to latent.
#     # Sample is B x C x T x H x W, while spectral_mask_sequence expects B x T x C x H x W, yet it works
#     # because internally it applies 2D FFT, i.e. only to the last two dimensions.
#     decoded_original: torch.Tensor = pipeline.decode(predicted_original)
#     # Spatially apply low and high-pass filtering to each frame and channel of the video.
#     masked_batches: list[dict[str, torch.Tensor]] = []
#     for i in range(decoded_original.size(1)):  # Apply separately over each channel C to reduce required memory.
#         # Sample is B x C x T x H x W, while spectral_mask_sequence expects B x T x C x H x W, yet it works
#         # because internally it applies 2D FFT, i.e. only to the last two dimensions.
#         masked_batch: dict[str, torch.Tensor] = masking_strategies.spectral_mask_sequence(
#             decoded_original[:, i].unsqueeze(dim=1).to(torch.float32), clamp_min=-1.0, clamp_max=1.0
#         )
#         masked_batches.append(masked_batch)
#     masked: dict[str, torch.Tensor] = {
#         n: torch.cat([m[n] for m in masked_batches], dim=1).to(predicted_original.dtype)
#         for n in masked_batches[0].keys()
#     }
#     masked = {n: pipeline.encode(t) for n, t in masked.items()}
#     # masked: dict[str, torch.Tensor] = masking_strategies.spectral_mask_sequence(
#     #     predicted_original, clamp_min=None, clamp_max=None
#     # )
#     low_pass: torch.Tensor = masked[f"spectral_low_{mask_radius}"]
#     high_pass: torch.Tensor = masked[f"spectral_high_{mask_radius}"]
#
#     if similarities_storage is not None:
#         similarities_to_compute: list[tuple[str, torch.Tensor, torch.Tensor]] = [
#             ("predicted_original_predicted_low_pass", predicted_original, predicted_low_pass),
#             ("predicted_original_predicted_high_pass", predicted_original, predicted_high_pass),
#             ("predicted_low_pass_predicted_high_pass", predicted_low_pass, predicted_high_pass),
#             ("predicted_low_pass_low_pass", predicted_low_pass, low_pass),
#             ("predicted_high_pass_high_pass", predicted_high_pass, high_pass),
#         ]
#         for similarity_label, t1, t2 in similarities_to_compute:
#             similarity: torch.Tensor = nn.functional.cosine_similarity(t1, t2, dim=-1)
#             similarity_mean: torch.Tensor = similarity.mean((-1, -2, -3))
#             similarity_std: torch.Tensor = similarity.std((-1, -2, -3))
#             similarities_storage.add_similarity(Similarity(
#                 file_path=pathlib.Path(output_path),
#                 step=step,
#                 label=f"{similarity_label}_{mask_radius}",
#                 similarity_mean=similarity_mean.detach().cpu().item(),
#                 similarity_std=similarity_std.detach().cpu().item(),
#             ))


class WhiteGaussianNoiseConsistencyLoss(nn.Module):
    def __init__(
        self,
        skewness_weight: float = 1.0,
        kurtosis_weight: float = 1.0,
        spectral_flatness_weight: float = 1.0,
        mean_weight: float = 1.0,
        variance_weight: float = 1.0,
        target_mean: float = 0.0,
        target_var: float = 1.0,
        low_pass_skewness_weight: float = .0,
        low_pass_kurtosis_weight: float = .0,
        sfm_dtype: str | None = None,
        loss_dtype: str | None = None,
        use_hann_window: bool = False
    ):
        super().__init__()
        self.target_mean: float = target_mean
        self.target_var: float = target_var
        self.weights: dict[str, float] = {
            "skew": skewness_weight,
            "kurt": kurtosis_weight,
            "sfm": spectral_flatness_weight,
            "mean": mean_weight,
            "var": variance_weight,
            "skew_low": low_pass_skewness_weight,
            "kurt_low": low_pass_kurtosis_weight
        }
        self.sfm_dtype: torch.dtype | None = getattr(torch, sfm_dtype) if sfm_dtype else None
        self.loss_dtype: torch.dtype | None = getattr(torch, loss_dtype) if loss_dtype else None
        self.use_hann_window: bool = use_hann_window

    def get_gaussian_scores(self, x: torch.Tensor):
        """
        Computes statistics per sample in the batch.
        Assumes x shape: (Batch, ...)
        """
        if self.loss_dtype is not None:
            x = x.to(self.loss_dtype)

        # Flatten all dims except batch: (B, N)
        B = x.size(0)
        flat = x.reshape(B, -1)

        # 1. Basic Stats (Mean & Var)
        # dim=1 calculates stat for each sample individually
        mean = torch.mean(flat, dim=1)
        var = torch.var(flat, dim=1, unbiased=False)  # unbiased=False matches standard moment def
        std = torch.sqrt(var + 1e-8)

        # 2. Higher Order Moments (Skew & Kurtosis)
        # z_scores shape: (B, N)
        # We must use keepdim=True for mean/std to broadcast correctly
        z_scores = (flat - mean.unsqueeze(1)) / std.unsqueeze(1)

        # Skewness = E[z^3], Kurtosis = E[z^4] - 3.
        # Integer powers expanded by hand to avoid the generic torch.pow path.
        z2 = z_scores * z_scores
        skew = torch.mean(z2 * z_scores, dim=1)
        kurtosis = torch.mean(z2 * z2, dim=1) - 3.0

        # 3. Spectral Flatness (Target 1.0)
        # We calculate this per sample as well.
        # Note: fft2 expects last 2 dims to be spatial.
        if self.sfm_dtype is None:
            fft_dtype: torch.dtype = torch.float32
            sfm_dtype: torch.dtype = x.dtype
        else:
            fft_dtype: torch.dtype = self.sfm_dtype
            sfm_dtype: torch.dtype = self.sfm_dtype

        # # Apply Windowing to prevent Spectral Leakage
        # if self.use_hann_window:
        #     B, T, C, H, W = x.size()
        #
        #     # # This ensures DC energy doesn't leak into other bins when windowing
        #     # x = x - x.mean(dim=(3, 4), keepdim=True)
        #
        #     # Create 2D Hanning window
        #     win_y = torch.hann_window(H, device=x.device)
        #     win_x = torch.hann_window(W, device=x.device)
        #     # Outer product to make 2D window
        #     window_2d = win_y.unsqueeze(1) * win_x.unsqueeze(0)  # (H, W)
        #     # Broadcast to batch/channel
        #     x = x * window_2d.view(1, 1, 1, H, W)
        #
        # if x.ndim >= 3:
        #     fft = torch.fft.fft2(x.to(fft_dtype))
        # else:
        #     # Fallback for 1D/2D tensors
        #     fft = torch.fft.fft(x)
        #
        # psd = torch.abs(fft) ** 2
        #
        # # Flatten PSD per sample: (B, -1)
        # psd_flat = psd.reshape(B, -1)
        #
        # # Mask out DC component (index 0) to avoid log(0) issues if mean is 0
        # # This takes everything from index 1 onwards
        # psd_no_dc = psd_flat[:, 1:] + 1e-10
        # psd_no_dc = psd_no_dc.to(sfm_dtype)
        #
        # geom_mean = torch.exp(torch.mean(torch.log(psd_no_dc), dim=1))
        # arith_mean = torch.mean(psd_no_dc, dim=1)
        #
        # # Prevent division by zero if signal is silent
        # sfm = geom_mean / (arith_mean + 1e-10)

        # Reshape to isolate spatial planes: (N, H, W) where N = B*T*C
        # We treat every frame/channel as an independent image for FFT
        x_spatial = x.reshape(-1, x.size(-2), x.size(-1))
        N, H, W = x_spatial.shape

        # A. PRE-WINDOW MEAN SUBTRACTION (CRITICAL)
        # We must zero-center the signal BEFORE windowing to avoid "pulse" artifacts
        x_spatial = x_spatial - x_spatial.mean(dim=(1, 2), keepdim=True)

        # B. Apply Windowing
        if self.use_hann_window:
            win_y = torch.hann_window(H, device=x.device)
            win_x = torch.hann_window(W, device=x.device)
            window_2d = win_y.unsqueeze(1) * win_x.unsqueeze(0)  # (H, W)
            # Broadcast to N samples
            x_spatial = x_spatial * window_2d.unsqueeze(0)

        # C. FFT & PSD. `|fft|**2` is computed directly from the real/imag
        # components to avoid the abs-then-square round-trip (sqrt + power).
        fft = torch.fft.fft2(x_spatial.to(fft_dtype))
        psd = fft.real * fft.real + fft.imag * fft.imag

        # D. AVERAGE PSD ACROSS BATCH (The Magic Step)
        # Instead of SFM per sample, we average the spectra first.
        # This smooths out the "jaggedness" of random noise.
        avg_psd = torch.mean(psd, dim=0)  # Shape: (H, W)

        # E. Compute SFM on the Averaged Spectrum
        psd_flat = avg_psd.flatten()

        # Remove DC (Index 0)
        psd_no_dc = psd_flat[1:] + 1e-10
        psd_no_dc = psd_no_dc.to(sfm_dtype)

        geom_mean = torch.exp(torch.mean(torch.log(psd_no_dc)))
        arith_mean = torch.mean(psd_no_dc)

        # This single scalar represents the "Whiteness" of the entire batch distribution
        sfm_score = geom_mean / (arith_mean + 1e-10)

        # Expand to batch size just to match return signature (optional)
        sfm = sfm_score.expand(B)
        return mean, var, skew, kurtosis, sfm

    def get_low_pass_gaussian_scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Use simple average pooling to approximate a low-pass filter.
        x = einops.rearrange(x, "b c t h w -> b (c t) h w")  # prepare input for avg_pool2d
        x = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # Flatten all dims except batch: (B, N)
        B = x.size(0)
        flat = x.reshape(B, -1)

        # 1. Basic Stats (Mean & Var)
        # dim=1 calculates stat for each sample individually
        mean = torch.mean(flat, dim=1)
        var = torch.var(flat, dim=1, unbiased=False)  # unbiased=False matches standard moment def
        std = torch.sqrt(var + 1e-8)

        # 2. Higher Order Moments (Skew & Kurtosis)
        # z_scores shape: (B, N)
        # We must use keepdim=True for mean/std to broadcast correctly
        z_scores = (flat - mean.unsqueeze(1)) / std.unsqueeze(1)

        # Skewness = E[z^3], Kurtosis = E[z^4] - 3.
        z2 = z_scores * z_scores
        skew = torch.mean(z2 * z_scores, dim=1)
        kurtosis = torch.mean(z2 * z2, dim=1) - 3.0

        return skew, kurtosis

    def forward(self, pred_noise: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        mean, var, skew, kurt, sfm = self.get_gaussian_scores(pred_noise)

        # Calculate Squared Errors
        loss_mean = (mean - self.target_mean) ** 2
        loss_var = (var - self.target_var) ** 2
        loss_skew = skew ** 2
        loss_kurt = kurt ** 2
        loss_sfm = (1.0 - sfm) ** 2

        # Average across the batch (mean reduction)
        total_loss = (
            self.weights["skew"] * torch.mean(loss_skew) +
            self.weights["kurt"] * torch.mean(loss_kurt) +
            self.weights["sfm"] * torch.mean(loss_sfm) +
            self.weights["mean"] * torch.mean(loss_mean) +
            self.weights["var"] * torch.mean(loss_var)
        )

        all_losses: dict[str, float] = {
            "loss_mean": torch.mean(loss_mean).item(),
            "loss_var": torch.mean(loss_var).item(),
            "loss_skew": torch.mean(loss_skew).item(),
            "loss_kurt": torch.mean(loss_kurt).item(),
            "loss_sfm": torch.mean(loss_sfm).item()
        }

        # Compute low-pass loss if such weights are > 0.
        if self.weights["skew_low"] > 0 or self.weights["kurt_low"] > 0:
            skew_low, kurt_low = self.get_low_pass_gaussian_scores(pred_noise)

            loss_skew_low = skew_low ** 2
            loss_kurt_low = kurt_low ** 2

            total_loss += (
                self.weights["skew_low"] * torch.mean(loss_skew_low) +
                self.weights["kurt_low"] * torch.mean(loss_kurt_low)
            )
            all_losses.update({
                "loss_skew_low": torch.mean(loss_skew_low).item(),
                "loss_kurt_low": torch.mean(loss_kurt_low).item()
            })

        return total_loss, all_losses


class SimilarityStorage:
    def __init__(self) -> None:
        self.storage: dict[pathlib.Path, dict[str, Any]] = {}

    def add_similarity(self, similarity: 'Similarity') -> None:
        self.storage[similarity.file_path] = self.storage.get(
            similarity.file_path, {"file": str(similarity.file_path)}
        )
        file_similarities: dict[str, Any] = self.storage[similarity.file_path]
        file_similarities[f"{similarity.label}_step_{similarity.step}_mean"] = similarity.similarity_mean
        file_similarities[f"{similarity.label}_step_{similarity.step}_std"] = similarity.similarity_std

    def save(self, out_path: pathlib.Path) -> Optional[pathlib.Path]:
        lock_file: pathlib.Path = out_path.parent / f"{out_path.name}.lock"
        lock = filelock.FileLock(lock_file)

        backup_file: Optional[pathlib.Path] = None
        with lock:
            if out_path.exists():
                self.load(out_path)
                backup_file = out_path.parent / f"{out_path.name}.bak"
                shutil.copyfile(out_path, backup_file)
            fieldnames: set[str] = set()
            for v in self.storage.values():
                fieldnames.update(v.keys())
            csv_tools.write_csv_file(list(self.storage.values()), output_file=out_path,
                                     fieldnames=list(fieldnames))

        return backup_file

    def load(
        self,
        csv_file: pathlib.Path,
        file_column: str = "file",
        delimiter: str = ","
    ) -> None:
        """Loads the contents of a CSV file into the current similarity storage.

        The content already in the current similarity storage overwrites any duplicate
        content loaded from the CSV file.

        This CSV file is expected to be a valid export of a similarity storage.
        No checks are performed to validate it.
        """
        csv_entries: list[dict[str, Any]] = csv_tools.read_csv_file(csv_file, delimiter=delimiter)
        for e in csv_entries:
            file_path: pathlib.Path = pathlib.Path(e[file_column])
            # The intention is that any duplicate entries existing in the current similarity
            # storage to always overwrite the previous ones in the loaded file. This way,
            # the most recently computed values are retained.
            previous_entries: Optional[dict[str, Any]] = self.storage.get(file_path, None)
            self.storage[file_path] = copy.deepcopy(e)
            if previous_entries is not None:
                self.storage[file_path].update(previous_entries)


@dataclasses.dataclass
class Similarity:
    file_path: pathlib.Path
    step: int
    label: str
    similarity_mean: float
    similarity_std: float