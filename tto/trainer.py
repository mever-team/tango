import copy
import logging
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from utils.distributed import (
    # EMA_FSDP,
    fsdp_wrap,
    # fsdp_state_dict,
    launch_distributed_job
)
from utils.misc import set_seed
from . import losses, lora


logger: logging.Logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.tto_epochs = config.tto.epochs
        self.loss = config.tto.loss
        self.loss_fn = None
        self.use_lora: bool = config.tto.get("use_lora", False)
        self.lora_rank: int = config.tto.get("lora_rank", 4)
        self.lora_chain_per_n_blocks: int | None = config.tto.get("lora_chain_per_n_blocks", None)
        self.lora_modules: list[str] = config.tto.get("lora_modules", [".self_attn.q", ".self_attn.k",
                                                                       ".self_attn.v", ".self_attn.o"])
        self._init_loss_fn()

        self.masking_radius = config.tto.masking_radius

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        self.global_rank = dist.get_rank()
        self.local_rank = dist.get_node_local_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.device(f"cuda:{self.local_rank}")

        # If None/negative seed is passed initialize a random one. Otherwise, use the provided one for reproducibility.
        if config.seed is None or int(config.seed) < 0:
            config.seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
            print(f"[seed] randomly chosen master seed: {config.seed}", flush=True)
        else:
            config.seed = int(config.seed)
            print(f"[seed] master seed: {config.seed}", flush=True)
        set_seed(config.seed)

        self.model = None
        self.critic_model = None
        self.optimizer = None
        self.fsdp_state_dict_backup = None
        self.unwrapped_model_backup = None

    def _init_loss_fn(self) -> None:
        if self.loss == "gaussian_forcing":
            logger.info("Initializing gaussian forcing loss.")
            gaussian_forcing_loss_params: dict[str, Any] | None = self.config.tto.get("gaussian_forcing_loss", None)
            if gaussian_forcing_loss_params is not None:
                logger.info(str(gaussian_forcing_loss_params))
                self.loss_fn = losses.WhiteGaussianNoiseConsistencyLoss(**gaussian_forcing_loss_params)
            else:
                self.loss_fn = losses.WhiteGaussianNoiseConsistencyLoss()
        else:
            raise NotImplementedError(f"Unsupported loss: {self.loss}")

    def _init_optimizer(self) -> None:
        print(f"Total parameters: {count_parameters(self.model, trainable=False)}")
        print(f"Trainable parameters: {count_parameters(self.model)}")
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
            fused=True
        )

    def set_model(self, model) -> None:
        """Sets the model that will be used for test time optimization."""
        if self.config.tto.get("critic_model", None) == "init":
            # When the initial model, i.e. not the TTOed, should be used as a critic, a frozen
            # copy of the model is required to remain a memory. The frozen is used as the critic,
            # while the non-frozen one gets updated at test time.
            self.critic_model = copy.deepcopy(model).eval().requires_grad_(False)

        if self.use_lora:
            lora.inject_lora(model.model, rank=self.lora_rank, target_modules=self.lora_modules)
            # Retain the original model, as editing the one in FSDP is not possible.
            self.unwrapped_model_backup = copy.deepcopy(model).to("cpu")

        self.model = self._wrap_for_distribution(
            model, cpu_offload=self.config.tto.cpu_offload_actor
        )
        if self.config.tto.get("critic_model", None) == "init":
            self.critic_model = self._wrap_for_distribution(
                self.critic_model, cpu_offload=self.config.tto.cpu_offload_critic
            )
        if self.config.tto.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
            if self.critic_model:
                self.critic_model.enable_gradient_checkpointing()

        self._init_optimizer()

        self.fsdp_state_dict_backup = self._snapshot_state_dict(
            self.model, device=torch.device("cpu")
        )

        # def detect_nan_grad(name):
        #     def hook(grad):
        #         if torch.isnan(grad).any() or torch.isinf(grad).any():
        #             print(f"NaN or Inf gradient detected in: {name}")
        #         return grad
        #
        #     return hook
        #
        # for name, p in self.model.named_parameters():
        #     p.register_hook(detect_nan_grad(name))

    def chain_lora(self) -> None:
        if not self.use_lora:
            raise RuntimeError("Chaining LoRA requires LoRA training to be enabled.")

        # Update the state dict of the unwrapped backup model with the trained one.
        trained_state_dict: dict = self._snapshot_state_dict(
            self.model, device=torch.device("cpu")
        )
        self.unwrapped_model_backup.load_state_dict(trained_state_dict)

        # Inject an additional LoRA adapter to the backup model.
        lora.inject_chain_lora(self.unwrapped_model_backup.model, rank=self.lora_rank, target_modules=self.lora_modules)
        model = copy.deepcopy(self.unwrapped_model_backup)

        # Rewrap (FSDP if multi-rank, otherwise just place on device).
        del self.model
        self.model = self._wrap_for_distribution(
            model, cpu_offload=self.config.tto.cpu_offload_actor
        )

        # Re-init optimizer with the new trainable params.
        self._init_optimizer()

    def backward_loss(self, loss: torch.Tensor):
        # print(torch.cuda.memory_allocated() / 1024 ** 2, "Before backward() MB allocated")
        # print(torch.cuda.memory_reserved() / 1024 ** 2, "Before backward() MB reserved")
        # loss.retain_grad()
        loss.backward()
        # print(torch.cuda.memory_allocated() / 1024 ** 2, "Before step() MB allocated")
        # print(torch.cuda.memory_reserved() / 1024 ** 2, "Before step () MB reserved")
        self.optimizer.step()
        # print(torch.cuda.memory_allocated() / 1024 ** 2, "After step() MB allocated")
        # print(torch.cuda.memory_reserved() / 1024 ** 2, "After step () MB reserved")

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    # def step(self) -> None:
    #     self.optimizer.step()

    def reset_model(self) -> None:
        if self.use_lora:
            # # Remove all previous LoRA layers.
            # lora.remove_lora(self.unwrapped_model_backup.model)
            # # Insert newly initialized LoRA layers.
            # lora.inject_lora(self.unwrapped_model_backup.model, rank=self.lora_rank, target_modules=self.lora_modules)
            # model = copy.deepcopy(self.unwrapped_model_backup)
            # # Rewrap in FSDP.
            # del self.model
            # self.model = fsdp_wrap(
            #     model,
            #     sharding_strategy=self.config.sharding_strategy,
            #     mixed_precision=self.config.mixed_precision,
            #     wrap_strategy=self.config.generator_fsdp_wrap_strategy,
            #     cpu_offload=self.config.tto.cpu_offload_actor,
            #     min_num_params=int(5e7)
            # )
            # # Re-init the optimizer with the new trainable LoRA parameters.
            # self._init_optimizer()
            lora.reset_lora_in_place(self.model)
            self.optimizer.state.clear()
        else:
            self._restore_state_dict(self.model, self.fsdp_state_dict_backup)
            self.zero_grad()
            self.optimizer.state.clear()

    def augment_input(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # FSDP-only helpers. When `world_size == 1`, FSDP wrapping adds an
    # NCCL all-reduce per backward with no sharding benefit (and it is the
    # path that triggered the random `Failed to CUDA calloc async ...`
    # NCCL crash). In that regime we keep the bare module on the trainer
    # device and use plain `state_dict()` / `load_state_dict(...)`.
    # ------------------------------------------------------------------

    def _wrap_for_distribution(
        self,
        module: torch.nn.Module,
        cpu_offload: bool,
    ) -> torch.nn.Module:
        if self.world_size > 1:
            return fsdp_wrap(
                module,
                sharding_strategy=self.config.sharding_strategy,
                mixed_precision=self.config.mixed_precision,
                wrap_strategy=self.config.generator_fsdp_wrap_strategy,
                cpu_offload=cpu_offload,
                min_num_params=int(5e7),
            )
        return module.to(self.device)

    def _snapshot_state_dict(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> dict:
        if self.world_size > 1:
            return deepcopy_sharded_state_dict(model, device=device)
        return {
            k: v.detach().clone().to(device)
            for k, v in model.state_dict().items()
        }

    def _restore_state_dict(
        self,
        model: torch.nn.Module,
        state_dict: dict,
    ) -> None:
        if self.world_size > 1:
            restore_sharded_state_dict(model, state_dict)
        else:
            model.load_state_dict(state_dict)


def deepcopy_sharded_state_dict(fsdp_model, device: torch.device) -> dict:
    # Enter the sharded context.
    with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
        # Get the state dict (returns local shards only).
        state_dict = fsdp_model.state_dict()

        # Clone the tensors so they don't update during training.
        copied_state_dict = {k: v.clone().detach().to(device) for k, v in state_dict.items()}

    return copied_state_dict


def restore_sharded_state_dict(fsdp_model, sharded_state_dict: dict) -> None:
    # Enter the sharded contex.
    with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
        # Load the snapshot. If moved to CPU earlier, FSDP handles the move back to GPU automatically.
        fsdp_model.load_state_dict(sharded_state_dict)


def count_parameters(model, trainable: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable)
