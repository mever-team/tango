import numpy as np
import random
import torch


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict


def seed_for_video(master_seed: int, video_idx: int) -> int:
    """Derive a per-video RNG seed from the run's master seed and the dataset index.

    Why this exists: the per-video sampling noise used to be drawn from the global
    generator inside the video loop, so it depended on how many random numbers had
    already been consumed. Any change to the TTO configuration (number of epochs,
    LoRA resets, look-ahead draws) shifts that stream, and every *subsequent* video
    of the run then gets different noise. Two arms that differ only in their TTO
    settings therefore generate different videos for reasons unrelated to the
    treatment, which makes them unpairable.

    Seeding from (master_seed, dataset idx) instead makes the trajectory noise a
    pure function of the run seed and the video, identical across arms.

    :param master_seed: the run's master seed (`config.seed`).
    :param video_idx: the dataset index of the video (stable across resume/skip).
    :returns: a seed valid for `torch.Generator.manual_seed`.
    """
    return (int(master_seed) * 1_000_003 + int(video_idx)) % (2 ** 31 - 1)
