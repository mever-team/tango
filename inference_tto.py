import os
import argparse
import pathlib
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils import csv_tools
from utils.dataset import TextDataset, TextImagePairDataset, TextVideoPairDataset
from utils.misc import seed_for_video, set_seed
from tto import trainer
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--csv_root_dir", type=str,
                    help="Applicable only when `--data_path` is a CSV file. Path of the directory "
                         "to which the paths defined in the CSV are relative.")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--num_input_latent_frames", type=int, default=3)
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--v2v", action="store_true", help="Whether to perform V2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=None,
                    help="Master RNG seed. Overrides the config. Use -1 for a random (but logged) seed.")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--visual_prompt_end_frame", type=int, default=93,
                    help="Index of the last frame (1-indexed) in input videos that can be used "
                         "as a visual prompt. Input RGB frames will be sampled before this frame "
                         "(including it). Relevant only on V2V generation.")
parser.add_argument("--tag", type=str, default="self_forcing_dmd",
                    help="The tag to identify this run.")
parser.add_argument("--tto_losses", action="store_true",
                     help="Computes and exports to CSV the candidate TTO losses.")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="Override config fields, e.g. `--set tto.epochs=10 tto.batch_size=2`. "
                         "Applied after merging the default and provided configs.")
args = parser.parse_args()

# Load config.
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
if args.set:
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.set))
# `--seed` used to be parsed and never read. Wire it in before the Trainer is
# built, since that is where the master seed is consumed.
if args.seed is not None:
    config.seed = args.seed

# Initialize distributed TTO trainer.
tto_trainer: trainer.Trainer = trainer.Trainer(config)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory: bool = get_cuda_free_memory_gb(gpu) < 80

# Initialize pipeline.
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=tto_trainer.device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=tto_trainer.device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

pipeline = pipeline.to(dtype=tto_trainer.dtype)
if low_memory:  # TODO: Make sure this can work under FSDP, otherwise remove it.
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=tto_trainer.device)
else:
    pipeline.text_encoder.to(device=tto_trainer.device)
pipeline.vae.to(device=tto_trainer.device)  # If FSDP on VAE is enabled, comment out this. Yet, FSDP seems to cause issues.
tto_trainer.set_model(pipeline.generator)

# Create dataset. It should be the same among all FSDP processes.
assert not args.i2v or not args.v2v, "Only one of --i2v or --v2v arguments should be provided."
if args.i2v or args.v2v:
    if args.i2v:
        transform = transforms.Compose([
            transforms.Resize((480, 832)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = TextImagePairDataset(args.data_path, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize((480, 832)),
            transforms.Normalize([0.5], [0.5])
        ])
        # 3 RGB frames per latent frame was a value initially used, despite the fact that wan
        # encoder supports 4 such frames. So, if not specified differently, keeping this previous
        # value as default, makes it compatible with these initial experiments.
        # TODO: Evaluate whether this number makes any difference and possibly change it.
        rgb_frames_per_latent_frame: int = config.get("rgb_frames_per_latent_frame", 3)
        num_input_rgb_frames: int = args.num_input_latent_frames * rgb_frames_per_latent_frame
        dataset = TextVideoPairDataset(
            pathlib.Path(args.data_path),
            csv_root_dir=pathlib.Path(args.csv_root_dir),
            transform=transform,
            frames_num=num_input_rgb_frames,
            prompt_column="caption",
            end_frame=args.visual_prompt_end_frame
        )
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions).
if tto_trainer.local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

# A CSV will be exported to the output directory.
csv_filename: str = f"{pathlib.Path(args.data_path).stem}_{args.tag}.csv"
csv_output_path: pathlib.Path = pathlib.Path(args.output_folder) / csv_filename
out_csv_entries: list[dict[str, Any]] = []
if csv_output_path.exists():
    out_csv_entries = csv_tools.read_csv_file(csv_output_path)
    print("Continuing from previous CSV...")
    print(f"Videos already generated: {len(out_csv_entries)}")
    # Fix missing entries for backwards compatibility.
    for e in out_csv_entries:
        e["sample_num"] = e.get("sample_num", 0)

for i, batch_data in tqdm(enumerate(dataloader), disable=(tto_trainer.local_rank != 0)):
    idx = batch_data['idx'].item()

    # Per-video RNG isolation: seed from (master seed, dataset idx) so the
    # trajectory noise is a pure function of the run seed and the video, not of how
    # many random draws earlier videos consumed. See utils.misc.seed_for_video.
    video_seed = seed_for_video(config.seed, idx)
    set_seed(video_seed)
    noise_gen = torch.Generator(device=tto_trainer.device)
    noise_gen.manual_seed(video_seed)

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    with torch.no_grad():
        if args.i2v:
            # For image-to-video, batch contains image and caption
            prompt = batch['prompts'][0]  # Get caption from batch
            prompts = [prompt] * args.num_samples

            # Process the image
            image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(
                device=tto_trainer.device, dtype=tto_trainer.dtype)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(
                device=tto_trainer.device, dtype=tto_trainer.dtype)
            initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames - 1, 16, 60, 104],
                generator=noise_gen,
                device=tto_trainer.device, dtype=tto_trainer.dtype
            )
        elif args.v2v:
            # For image-to-video, batch contains image and caption
            prompt = batch['prompts'][0]  # Get caption from batch
            prompts = [prompt] * args.num_samples

            # Process the video.
            video = batch['video'].permute(0,2,1,3,4).to(
                device=tto_trainer.device, dtype=tto_trainer.dtype)  # B x C x T x H x W

            # Encode the input video as the first latents.
            initial_latent = pipeline.vae(video, mode="encode").to(device=tto_trainer.device, dtype=tto_trainer.dtype)
            # DEBUG CODE
            # d = pipeline.vae(initial_latent, mode="decode", use_cache=False)
            # d = d.detach().cpu().permute(0, 1, 3, 4, 2)
            # d = (d * 0.5 + 0.5).clamp(0, 1)
            # d = d * 255
            # write_video("debug.mp4", d.squeeze(dim=0), fps=16)
            initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames - initial_latent.size(1), 16, 60, 104],
                generator=noise_gen,
                device=tto_trainer.device, dtype=tto_trainer.dtype
            )
        else:
            # For text-to-video, batch is just the text prompt
            prompt = batch['prompts'][0]
            extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
            if extended_prompt is not None:
                prompts = [extended_prompt] * args.num_samples
            else:
                prompts = [prompt] * args.num_samples
            initial_latent = None

            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104],
                generator=noise_gen,
                device=tto_trainer.device, dtype=tto_trainer.dtype
            )

        # Sync the random noise across all ranks.
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast(sampled_noise, src=0)

    # If all the output files for the requested number of samples exist, skip generation.
    model_name: str = "regular" if not args.use_ema else "ema"
    if args.save_with_index:
        output_path_template: str = os.path.join(
            args.output_folder, f'{idx}-{{}}_{model_name}.mp4'
        )
    else:
        # Sanitize the prompt to only contain valid filename chars under a linux filesystem.
        filename_prompt: str = prompt[:100]
        filename_prompt = filename_prompt.replace("/", "_")
        filename_prompt = filename_prompt.replace("\0", "_")
        data_path: pathlib.Path = pathlib.Path(args.data_path)
        if data_path.is_file() and data_path.suffix == ".csv":
            output_path_template: str = os.path.join(
                args.output_folder, data_path.stem, str(args.tag),
                f'{idx}-{filename_prompt}-{{}}.mp4'
            )
        else:
            output_path_template: str = os.path.join(
                args.output_folder, str(args.tag), f'{idx}-{filename_prompt}-{{}}.mp4'
            )

    # Skip already generated videos by checking CSV entries.
    for seed_idx in range(args.num_samples):
        output_path: str = output_path_template.format(seed_idx)
        relative_output_path: pathlib.Path = pathlib.Path(output_path).relative_to(args.output_folder)
        # TODO: Make the search more efficient, as it currently iterates each time over the entire csv.
        generated_videos: list[pathlib.Path] = [pathlib.Path(e["gen_video"]) for e in out_csv_entries]
        if not relative_output_path in generated_videos:
            # If at least the file for a sample is not present, then all the samples
            # will be generated and any existing ones will be overridden. If this
            # creates any issues, may be changed in the future.
            break
    else:
        for seed_idx in range(args.num_samples):
            print(f"Skipping the generation of '{output_path_template.format(seed_idx)}'. "
                  "File exists.")
        continue

    # Generate frames (81 RGB for 21 latent frames).
    if args.v2v:
        # with torch.utils.checkpoint.set_checkpoint_debug_enabled(True):
        _, latents, videos_per_epoch = pipeline.inference_train_tto(
            tto_trainer=tto_trainer,
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_rgb=video,
            low_memory=low_memory,
        )
    elif args.i2v:
        _, latents, videos_per_epoch = pipeline.inference_train_tto(
            tto_trainer=tto_trainer,
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_rgb=image,
            low_memory=low_memory,
        )
    else:
        _, latents, videos_per_epoch = pipeline.inference_train_tto(
            tto_trainer=tto_trainer,
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_rgb=initial_latent,
            low_memory=low_memory,
        )
    tto_trainer.reset_model()

    # Clear VAE cache.
    pipeline.vae.model.clear_cache()

    # Export videos. Only rank 0 writes to storage.
    if tto_trainer.local_rank == 0:
        for video_data in videos_per_epoch.values():
            video = rearrange(video_data["video"], 'b t c h w -> b t h w c').detach().cpu()
            # all_video.append(current_video)
            # num_generated_frames += latents.shape[1]

            # Final output video
            video = 255.0 * video # * torch.cat(video, dim=1)

            # Save the video if the current prompt is not a dummy prompt
            if idx < num_prompts:
                for seed_idx in range(args.num_samples):
                    if video_data["label"] == "original" and video_data["tto_epoch"] == "final":
                        output_path: str = output_path_template.format(seed_idx)
                    else:
                        output_path: str = output_path_template.format(
                            f"{seed_idx}_tto_epoch_{video_data['tto_epoch']}_{video_data['label']}"
                        )

                    # Create the structure of the output CSV for new entries.
                    relative_output_path: pathlib.Path = pathlib.Path(output_path).relative_to(args.output_folder)
                    if args.v2v:
                        csv_entry: dict[str, Any] = {
                            "video": str(dataset.get_video_path(idx).relative_to(args.csv_root_dir)),
                            "text_prompt": prompt,
                            "visual_prompt_start": args.visual_prompt_end_frame - num_input_rgb_frames,
                            "visual_prompt_end": args.visual_prompt_end_frame - 1,
                            "gen_video": relative_output_path,
                            "gen_tag": args.tag,
                            "gen_fps": 16,
                            "gen_visual_prompt_start": 0,
                            "gen_visual_prompt_end": num_input_rgb_frames - 1,
                            "gen_visual_prompt_fps": batch["fps"].item(),
                            "tto_epoch": video_data["tto_epoch"],
                            "modifier": video_data["label"],
                            "sample_num": seed_idx
                        }
                    else:
                        csv_entry: dict[str, Any] = {
                            "text_prompt": prompt,
                            "gen_video": relative_output_path,
                            "gen_tag": args.tag,
                            "gen_fps": 16,
                            "tto_epoch": video_data["tto_epoch"],
                            "modifier": video_data["label"],
                            "sample_num": seed_idx
                        }
                    out_csv_entries.append(csv_entry)

                    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    write_video(output_path, video[seed_idx], fps=16)

        csv_tools.write_csv_file(out_csv_entries, csv_output_path)

# def save_video(video: torch.Tensor, video_path: pathlib.Path, fps: int = 16) -> None:
#     video = 255.0 * torch.cat(all_video, dim=1)
#     pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
#     write_video(output_path, video[seed_idx], fps=16)