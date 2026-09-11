import os
import pathlib
import json
from typing import Optional, Any, Callable

import numpy as np
import torch
import lmdb
import decord
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from utils import csv_tools


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


class TextVideoPairDataset(Dataset):
    def __init__(
        self,
        csv_path: pathlib.Path,
        transform: Callable = None,
        frames_num: Optional[int] = None,
        eval_first_n: Optional[int] = None,
        csv_root_dir: Optional[pathlib.Path] = None,
        csv_delimiter: str = ",",
        video_path_column: str = "video",
        prompt_column: str = "prompt",
        end_frame: Optional[int] = None,
        target_fps: Optional[float] = None,
    ):
        # Load video entries from the CSV file.
        self.entries: list[dict[str, Any]] = csv_tools.read_csv_file(
            csv_path, delimiter=csv_delimiter
        )
        if eval_first_n:
            self.entries = self.entries[:eval_first_n]

        self.csv_root_dir: Path = csv_root_dir or csv_path.parent
        self.video_path_column = video_path_column
        self.prompt_column: str = prompt_column
        self.transform: Callable = transform
        self.frames_num: Optional[int] = frames_num
        self.end_frame: Optional[int] = end_frame
        self.target_fps: Optional[float] = target_fps

        # Verify that all videos exist.
        for e in self.entries:
            video_path: Path = self.csv_root_dir / e[self.video_path_column]
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - video: torch.Tensor Video tensor with shape (T, C, H, W).
                - caption: str
                - idx: int Index of the video in the dataset.
        """
        video_path: pathlib.Path = self.get_video_path(idx)

        try:
            # Initialize VideoReader
            # ctx=cpu(0) is crucial for dataloaders to avoid CUDA context initialization errors
            # inside multiprocessing workers.
            vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
            src_fps: float = vr.get_avg_fps()

            end_frame: int = self.end_frame or len(vr)
            if end_frame >= self.frames_num:
                # If video is long enough take the last N frames.
                indices: list[int] = list(range(end_frame - self.frames_num, end_frame))
            else:
                # If video is shorter than N, repeat the first frame.
                pad_length: int = self.frames_num - end_frame
                indices: list[int] = [0] * pad_length + list(range(end_frame))

            # Efficient Batch Decoding
            # get_batch is significantly faster than looping through frames.
            frames = vr.get_batch(indices).asnumpy()  # Returns (T, H, W, C)

            # Convert to PyTorch Tensor. Decord returns (T, H, W, C) by default.
            frames_tensor: torch.Tensor = (
                    torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0)  # (T, C, H, W)
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            raise e

        if self.transform:
            frames_tensor = self.transform(frames_tensor)

        return {
            "video": frames_tensor,
            "prompts": self.get_prompt(idx),
            "idx": idx,
            "fps": src_fps
        }

    def get_video_path(self, idx: int) -> pathlib.Path:
        return self.csv_root_dir / self.entries[idx][self.video_path_column]

    def get_prompt(self, idx: int) -> str:
        return self.entries[idx][self.prompt_column]


def cycle(dl):
    while True:
        for data in dl:
            yield data
