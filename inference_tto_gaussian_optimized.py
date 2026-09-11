"""V2V/T2V inference entry point for white-Gaussian-noise consistency TTO.

This is a focused subset of `inference_tto.py`. It exercises the
gaussian consistency loss with an `init` critic, and supports both
V2V (visual prompt video) and T2V (text prompt only) modes via the
`--v2v` flag.

Outputs:

    * Per-video CSV (`<data_stem>_<tag>.csv`): bookkeeping columns
      plus `gen_wall_clock_seconds` and sub-component timing breakdowns
      (`gen_text_encode_seconds`, `gen_vae_encode_seconds`,
       `gen_rollout_seconds`, `gen_vae_decode_seconds`). The V2V mode
      additionally records `visual_prompt_*` / `gen_visual_prompt_*` cols.
    * Sidecar JSON (`<data_stem>_<tag>.json`): a dict keyed by the per-video
      `gen_video` relative path. Each entry maps
      `block_index -> step_index -> epoch -> {loss_name: float}` for offline
      analysis. Stored separately from the CSV so the CSV stays readable.
"""

import argparse
import json
import os
import pathlib
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms
from tqdm import tqdm
import imageio

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
# Use the memory-optimised pipeline. Aliased to the same local name so the
# rest of this file is byte-identical to `inference_tto_gaussian.py`.
from pipeline.causal_inference_tto_optimized import (
    CausalInferenceTTOOptimizedPipeline as CausalInferenceTTOPipeline,
)
from tto import trainer
from utils import csv_tools
from utils.dataset import TextDataset, TextVideoPairDataset
from utils.misc import seed_for_video, set_seed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the checkpoint file")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset (V2V CSV or T2V prompt file)")
    parser.add_argument("--csv_root_dir", type=str, default=None,
                        help="Root dir to which paths in the dataset CSV are relative. "
                             "Required only when `--v2v` is set.")
    parser.add_argument("--extended_prompt_path", type=str, default=None,
                        help="Optional path to a per-line extended-prompt file used in T2V mode")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output folder")
    parser.add_argument("--num_output_frames", type=int, default=63,
                        help="Total number of latent frames to generate. In V2V mode this "
                             "INCLUDES the conditioning latent frames; in T2V mode it equals "
                             "the number of denoised frames.")
    parser.add_argument("--num_input_latent_frames", type=int, default=3,
                        help="Number of conditioning latent frames (V2V only)")
    parser.add_argument("--v2v", action="store_true",
                        help="Run in V2V (visual prompt) mode. Default is T2V.")
    parser.add_argument("--use_ema", action="store_true",
                        help="Whether to load EMA generator weights")
    parser.add_argument("--seed", type=int, default=None,
                        help="Master RNG seed. Overrides the config. Use -1 for a random (but logged) seed.")
    parser.add_argument("--visual_prompt_end_frame", type=int, default=93,
                        help="Index of the last RGB frame (1-indexed) used as visual prompt (V2V only)")
    parser.add_argument("--tag", type=str, default="self_forcing_dmd_tto_gaussian",
                        help="Tag identifying this run")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config fields, e.g. `--set tto.epochs=10 tto.batch_size=2`. "
                             "Applied after merging the default and provided configs.")
    return parser


def _load_config(config_path: str, overrides: list[str]):
    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(config_path),
    )
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    return config


def _build_dataset(args, config, num_input_rgb_frames: int):
    """Construct the V2V or T2V dataset based on `--v2v`."""
    if args.v2v:
        if args.csv_root_dir is None:
            raise ValueError("--csv_root_dir is required in --v2v mode.")
        transform = transforms.Compose([
            transforms.Resize((480, 832)),
            transforms.Normalize([0.5], [0.5]),
        ])
        return TextVideoPairDataset(
            pathlib.Path(args.data_path),
            csv_root_dir=pathlib.Path(args.csv_root_dir),
            transform=transform,
            frames_num=num_input_rgb_frames,
            prompt_column="caption",
            end_frame=args.visual_prompt_end_frame,
        )
    return TextDataset(
        prompt_path=args.data_path,
        extended_prompt_path=args.extended_prompt_path,
    )


def _output_path_for(args, idx: int, prompt: str) -> str:
    """Build the per-video output path."""
    filename_prompt: str = prompt[:100].replace("/", "_").replace("\0", "_")
    data_path: pathlib.Path = pathlib.Path(args.data_path)
    if data_path.is_file() and data_path.suffix == ".csv":
        return os.path.join(
            args.output_folder, data_path.stem, str(args.tag),
            f"{idx}-{filename_prompt}-0.mp4",
        )
    return os.path.join(
        args.output_folder, str(args.tag), f"{idx}-{filename_prompt}-0.mp4"
    )


def main() -> None:
    args = _build_arg_parser().parse_args()
    # Backwards-compat: V2V scripts predating the --v2v flag pass
    # --csv_root_dir but not --v2v. Treat --csv_root_dir as an
    # unambiguous V2V signal (T2V never sets it) and auto-enable V2V
    # mode. New scripts can still pass --v2v explicitly.
    if args.csv_root_dir is not None and not args.v2v:
        args.v2v = True
        print(
            "[info] --csv_root_dir was provided without --v2v; auto-enabling "
            "V2V mode. Pass --v2v explicitly to silence this notice."
        )
    config = _load_config(args.config_path, args.set)
    if args.seed is not None:  # TODO: Consider refactoring into a single config build function.
        config.seed = args.seed

    tto_trainer: trainer.Trainer = trainer.Trainer(config)
    print(f"Free VRAM {get_cuda_free_memory_gb(gpu)} GB")
    low_memory: bool = get_cuda_free_memory_gb(gpu) < 80

    pipeline = CausalInferenceTTOPipeline(config, device=tto_trainer.device)
    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        pipeline.generator.load_state_dict(
            state_dict["generator_ema" if args.use_ema else "generator"]
        )

    pipeline = pipeline.to(dtype=tto_trainer.dtype)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=tto_trainer.device)
    else:
        pipeline.text_encoder.to(device=tto_trainer.device)
    pipeline.vae.to(device=tto_trainer.device)
    tto_trainer.set_model(pipeline.generator)

    rgb_frames_per_latent_frame: int = config.get("rgb_frames_per_latent_frame", 3)
    num_input_rgb_frames: int = args.num_input_latent_frames * rgb_frames_per_latent_frame
    dataset = _build_dataset(args, config, num_input_rgb_frames)
    print(f"Number of prompts: {len(dataset)}")
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=SequentialSampler(dataset),
        num_workers=0, drop_last=False,
    )

    if tto_trainer.local_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    base_stem: str = f"{pathlib.Path(args.data_path).stem}_{args.tag}"
    csv_output_path: pathlib.Path = pathlib.Path(args.output_folder) / f"{base_stem}.csv"
    losses_output_path: pathlib.Path = pathlib.Path(args.output_folder) / f"{base_stem}.json"

    out_csv_entries: list[dict[str, Any]] = []
    if csv_output_path.exists():
        out_csv_entries = csv_tools.read_csv_file(csv_output_path)
        print(f"Continuing from previous CSV ({len(out_csv_entries)} videos already done).")

    out_losses: dict[str, Any] = {}
    if losses_output_path.exists():
        with losses_output_path.open("r") as f:
            out_losses = json.load(f)
        print(f"Loaded existing per-step losses for {len(out_losses)} videos.")

    already_generated: set[str] = {e["gen_video"] for e in out_csv_entries}

    for batch_data in tqdm(dataloader, disable=(tto_trainer.local_rank != 0)):
        batch = batch_data if isinstance(batch_data, dict) else batch_data[0]
        idx: int = batch["idx"].item()
        # Prefer the extended prompt for conditioning when one is provided.
        # Keep the raw prompt for the output filename and CSV bookkeeping.
        prompt: str = batch["prompts"][0]
        cond_prompt: str = prompt
        if not args.v2v and "extended_prompts" in batch:
            cond_prompt = batch["extended_prompts"][0]

        output_path: str = _output_path_for(args, idx, prompt)
        relative_output_path: pathlib.Path = pathlib.Path(output_path).relative_to(args.output_folder)
        if str(relative_output_path) in already_generated:
            print(f"Skipping the generation of '{output_path}'. File exists.")
            continue

        # Per-video RNG isolation: seed from (master seed, dataset idx) so the
        # trajectory noise is a pure function of the run seed and the video, not of
        # how many random draws earlier videos consumed. Without this, two arms
        # differing only in TTO settings sample different noise for every later
        # video and cannot be paired. See utils.misc.seed_for_video.
        video_seed: int = seed_for_video(config.seed, idx)
        set_seed(video_seed)
        noise_gen: torch.Generator = torch.Generator(device=tto_trainer.device)
        noise_gen.manual_seed(video_seed)

        # Sample noise + (V2V only) encode the conditioning video.
        with torch.no_grad():
            if args.v2v:
                video_in: Optional[torch.Tensor] = batch["video"].permute(0, 2, 1, 3, 4).to(
                    device=tto_trainer.device, dtype=tto_trainer.dtype
                )  # B x C x T x H x W
                initial_latent: torch.Tensor = pipeline.vae(video_in, mode="encode").to(
                    device=tto_trainer.device, dtype=tto_trainer.dtype
                )
                num_denoised_frames: int = args.num_output_frames - initial_latent.size(1)
            else:
                video_in = None
                # In T2V mode `--num_output_frames` is the number of denoised
                # frames; there are no conditioning latents to subtract.
                num_denoised_frames = args.num_output_frames
            sampled_noise: torch.Tensor = torch.randn(
                [1, num_denoised_frames, 16, 60, 104],
                generator=noise_gen,
                device=tto_trainer.device, dtype=tto_trainer.dtype,
            )
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.broadcast(sampled_noise, src=0)

        decoded_video, _, stats = pipeline.generate(
            tto_trainer=tto_trainer,
            noise=sampled_noise,
            text_prompts=[cond_prompt],
            initial_rgb=video_in,
            return_latents=False,
        )
        # Reset LoRA + clear VAE cache between videos.
        tto_trainer.reset_model()
        pipeline.vae.model.clear_cache()

        if tto_trainer.local_rank == 0:
            decoded_video = rearrange(decoded_video, "b t c h w -> b t h w c").detach()
            decoded_video = (decoded_video * 255.0).to(torch.uint8)

            csv_entry: dict[str, Any] = {
                "text_prompt": prompt,
                "gen_video": str(relative_output_path),
                "gen_tag": args.tag,
                "gen_fps": 16,
                "sample_num": 0,
                "gen_wall_clock_seconds": stats["wall_clock_seconds"],
                "gen_text_encode_seconds": stats["text_encode_seconds"],
                "gen_vae_encode_seconds": stats["vae_encode_seconds"],
                "gen_rollout_seconds": stats["rollout_seconds"],
                "gen_vae_decode_seconds": stats["vae_decode_seconds"],
            }
            if args.v2v:
                # V2V-only bookkeeping columns mirror inference_tto.py /
                # inference_tto_gaussian.py so the downstream metrics
                # pipelines accept the CSV unchanged.
                csv_entry.update({
                    "video": str(dataset.get_video_path(idx).relative_to(args.csv_root_dir)),
                    "visual_prompt_start": args.visual_prompt_end_frame - num_input_rgb_frames,
                    "visual_prompt_end": args.visual_prompt_end_frame - 1,
                    "gen_visual_prompt_start": 0,
                    "gen_visual_prompt_end": num_input_rgb_frames - 1,
                    "gen_visual_prompt_fps": batch["fps"].item(),
                })
            out_csv_entries.append(csv_entry)
            out_losses[str(relative_output_path)] = stats["per_step_losses"]
            already_generated.add(str(relative_output_path))

            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(output_path, decoded_video[0].cpu().numpy(), fps=16, codec="libx264")

            # ---- Optional viz output: per (block, subtimestep) MP4s. ----
            # The pipeline returns `stats["viz"]` as a dict mapping
            # `block_BB_subtimestep_S` -> uint8 [T, H, W, 3] RGB clip,
            # plus sidecar keys (`__epochs`, `__frames_per_epoch`,
            # `__metrics`, `__final_metrics`, `__diff`). Clip filenames
            # carry the LAST-chapter-vs-epoch-0 change metrics
            # (`_dmae=` in 0-255 units, zero-padded; `_dssim=` = 1-SSIM)
            # so the biggest-change clips are identifiable at a glance,
            # and `summary.csv` lists all groups sorted by change
            # magnitude for triage. `*_diff.mp4` twins hold the signed
            # difference overlays against the epoch-0 reference.
            viz = stats.get("viz")
            if viz:
                viz_dir = pathlib.Path(output_path).parent / "viz" / pathlib.Path(output_path).stem
                viz_dir.mkdir(parents=True, exist_ok=True)
                viz_metadata: dict[str, Any] = {}
                summary_rows: list[dict[str, Any]] = []
                base_keys = [k for k in viz if "__" not in k]
                for key in sorted(base_keys):
                    fm = viz.get(f"{key}__final_metrics") or {}
                    suffix = ""
                    if fm:
                        suffix = f"_dmae={fm['mae255']:06.2f}_dssim={fm['dssim']:.4f}"
                    clip = viz[key].cpu().numpy()  # uint8 [T, H, W, 3]
                    clip_name = f"{key}{suffix}.mp4"
                    imageio.mimwrite(str(viz_dir / clip_name), clip, fps=16, codec="libx264")
                    diff = viz.get(f"{key}__diff")
                    diff_name = None
                    if diff is not None:
                        diff_name = f"{key}{suffix}_diff.mp4"
                        imageio.mimwrite(
                            str(viz_dir / diff_name), diff.cpu().numpy(),
                            fps=16, codec="libx264",
                        )
                    viz_metadata[key] = {
                        "clip": clip_name,
                        "diff_clip": diff_name,
                        "epochs": viz.get(f"{key}__epochs", []),
                        "frames_per_epoch": viz.get(f"{key}__frames_per_epoch", -1),
                        "num_frames": int(clip.shape[0]),
                        "metrics_vs_epoch0": viz.get(f"{key}__metrics", []),
                    }
                    parts = key.split("_")  # block_BB_subtimestep_S
                    summary_rows.append({
                        "block": int(parts[1]),
                        "subtimestep": int(parts[3]),
                        "dmae255": round(fm.get("mae255", 0.0), 4),
                        "dssim": round(fm.get("dssim", 0.0), 6),
                        "p99_255": round(fm.get("p99_255", 0.0), 4),
                        "psnr": round(fm.get("psnr", 99.0), 2),
                        "clip": clip_name,
                        "diff_clip": diff_name or "",
                    })
                with (viz_dir / "metadata.json").open("w") as f:
                    json.dump(viz_metadata, f, indent=2)
                # Triage file: biggest TTO-induced changes first.
                summary_rows.sort(key=lambda r: r["dmae255"], reverse=True)
                csv_tools.write_csv_file(
                    summary_rows, viz_dir / "summary.csv",
                    fieldnames=["block", "subtimestep", "dmae255", "dssim",
                                "p99_255", "psnr", "clip", "diff_clip"],
                )

            # Reconcile column schema with previously loaded entries (which
            # may not have the new timing columns).
            fieldnames = list(csv_entry.keys())
            for e in out_csv_entries:
                for k in fieldnames:
                    e.setdefault(k, "")
            csv_tools.write_csv_file(out_csv_entries, csv_output_path, fieldnames=fieldnames)

            # Atomic write of the per-step losses sidecar so a crash mid-write
            # cannot leave behind a half-written file.
            tmp_losses_path: pathlib.Path = losses_output_path.with_suffix(".json.tmp")
            with tmp_losses_path.open("w") as f:
                json.dump(out_losses, f)
            tmp_losses_path.replace(losses_output_path)


if __name__ == "__main__":
    main()
