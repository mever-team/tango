import argparse
import pathlib
from typing import Any

import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils import csv_tools
from utils.dataset import TextDataset, TextImagePairDataset, TextVideoPairDataset
from utils.misc import set_seed

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
parser.add_argument("--seed", type=int, default=0, help="Random seed")
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

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
if args.set:
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.set))

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
assert not args.i2v or not args.v2v, "Only one of --i2v or --v2v arguments should be provided."
if args.i2v or args.v2v:
    assert not dist.is_initialized(), "I2V and V2V does not support distributed inference yet"
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

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output

# On video-to-video generation a CSV will be exported to the output directory.
out_csv_entries: list[dict[str, Any]] = []

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    elif args.v2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the video.
        video = batch['video'].permute(0,2,1,3,4).to(device=device, dtype=torch.bfloat16)  # B x C x T x H x W

        # Encode the input video as the first latents.
        initial_latent = pipeline.vae.encode_to_latent(video).to(device=device,
                                                                 dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - initial_latent.size(1), 16, 60, 104],
            device=device,
            dtype=torch.bfloat16
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
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

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
    for seed_idx in range(args.num_samples):
        output_path: str = output_path_template.format(seed_idx)

        if args.v2v:
            out_csv_entries.append({
                "video": str(dataset.get_video_path(idx).relative_to(args.csv_root_dir)),
                "text_prompt": prompt,
                "visual_prompt_start": args.visual_prompt_end_frame - num_input_rgb_frames,
                "visual_prompt_end": args.visual_prompt_end_frame - 1,
                "gen_video": pathlib.Path(output_path).relative_to(args.output_folder),
                "gen_tag": args.tag,
                "gen_fps": 16,
                "gen_visual_prompt_start": 0,
                "gen_visual_prompt_end": num_input_rgb_frames - 1,
                "gen_visual_prompt_fps": batch["fps"].item()
            })

        if not pathlib.Path(output_path).exists():
            # If at least the file for a sample is not present, then all the samples
            # will be generated and any existing ones will be overridden. If this
            # creates any issues, may be changed in the future.
            break
    else:
        for seed_idx in range(args.num_samples):
            print(f"Skipping the generation of '{output_path_template.format(seed_idx)}'. "
                  "File exists.")
        continue

    # Generate 81 frames
    tto_losses: dict[str, float] | None = None
    if args.v2v and args.tto_losses:
        video, latents, tto_losses = pipeline.inference_tto(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_rgb=video,
            low_memory=low_memory,
        )
    else:
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
            low_memory=low_memory,
        )
    if tto_losses is not None:
        out_csv_entries[-1].update(tto_losses)
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            output_path: str = output_path_template.format(seed_idx)
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            write_video(output_path, video[seed_idx], fps=16)

    if args.v2v and idx % 20 == 0:  # Save CSV every 20 iterations.
        csv_filename: str = f"{pathlib.Path(args.data_path).stem}_{args.tag}.csv"
        csv_output_path: pathlib.Path = pathlib.Path(args.output_folder) / csv_filename
        csv_tools.write_csv_file(out_csv_entries, csv_output_path)

if args.v2v:
    csv_filename: str = f"{pathlib.Path(args.data_path).stem}_{args.tag}.csv"
    csv_output_path: pathlib.Path = pathlib.Path(args.output_folder) / csv_filename
    csv_tools.write_csv_file(out_csv_entries, csv_output_path)
