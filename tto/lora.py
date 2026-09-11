from contextlib import contextmanager
from typing import Iterable, Iterator

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, original_layer: 'nn.Linear | LoRALinear', rank: int = 4):
        super().__init__()
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank

        # Freeze the original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # 1. Initialize Low-Rank Matrices
        # B: (Out_Features, Rank) - initialized to Zero
        # A: (Rank, In_Features)  - initialized to Kaiming/Random
        # This ensures that at step 0, LoRA output is 0 and the model behaves exactly like pretrained.
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # When True, `forward` bypasses the LoRA branch and returns the frozen
        # `original_layer(x)` directly — making the wrapped module bit-identical
        # to its pre-injection state. Toggled via the `disabled(...)` context
        # manager below; default behaviour is unchanged for every other caller.
        self.disable_lora: bool = False

    def forward(self, x):
        # 1. Original Path (Frozen)
        original_out = self.original_layer(x)

        if self.disable_lora:
            return original_out

        # 2. LoRA Path (Trainable)
        # Equation: Y = Wx + (B @ A)x * scale
        # We compute x @ A.T then result @ B.T to match Linear dimensions
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()

        return original_out + lora_out

    def reset(self) -> None:
        """Resets LoRA parameters in-place."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


@contextmanager
def disabled(model: nn.Module) -> Iterator[None]:
    """Temporarily skip the LoRA branch in every `LoRALinear` submodule of
    `model`. While inside the `with` block the model's forward is bit-identical
    to the pre-LoRA-injection original (i.e. equivalent to calling a frozen
    deep-copy of the generator made *before* `inject_lora`)."""
    layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    saved_flags = [m.disable_lora for m in layers]
    for m in layers:
        m.disable_lora = True
    try:
        yield
    finally:
        for m, prev in zip(layers, saved_flags):
            m.disable_lora = prev


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    target_modules: Iterable[str] = (".q", ".k", ".v", ".o")
) -> nn.Module:
    """
    Replaces Linear layers in 'model' with LoRALinear if their name matches target_modules.
    """
    model_dtype = model.dtype
    model_device = model.device

    # 1. Collect targets
    targets_to_replace = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if name matches targets
            if any(t in name for t in target_modules):
                # IMPORTANT: Check if we are already inside a LoRA layer
                # (e.g. accidentally matching 'lora.original_layer')
                parent_name = name.rsplit('.', 1)[0]
                parent = model.get_submodule(parent_name) if '.' in name else model
                if isinstance(parent, LoRALinear):
                    continue

                targets_to_replace.append((name, module))

    # 2. Apply modifications
    for name, module in targets_to_replace:
        parent_name = name.rsplit('.', 1)[0]
        child_name = name.rsplit('.', 1)[1]

        parent = model.get_submodule(parent_name) if '.' in name else model

        # Create LoRA wrapper
        lora_layer = LoRALinear(module, rank=rank)

        # Swap
        setattr(parent, child_name, lora_layer)
        print(f"Injecting LoRA: {name}")

    # Set requires_grad
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    model.to(model_dtype)
    model.to(model_device)

    return model


def inject_chain_lora(
    model: nn.Module,
    rank: int = 4,
    target_modules: Iterable[str] = (".q", ".k", ".v", ".o")
) -> nn.Module:
    """Chains LoRALinear if their name matches target_modules."""
    model_dtype = model.dtype
    model_device = model.device

    chained_lora_parameters_names: set[str] = set()

    for name, module in model.named_modules():
        # Check if this is a Linear layer and matches our target names
        if isinstance(module, LoRALinear):
            # Check if any target string is in the module name
            # e.g., "transformer.blocks.0.attn.q_proj"
            if any(t in name for t in target_modules):
                # Find the parent module to swap the child
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1]

                # Get parent object
                parent = model.get_submodule(parent_name) if '.' in name else model

                # In case of chained LoRA layers, wrapping should only be done to the outermost one.
                if isinstance(parent, LoRALinear):
                    continue

                # Create LoRA wrapper
                lora_layer = LoRALinear(module, rank=rank)

                # Swap it in place
                setattr(parent, child_name, lora_layer)

                print(f"Injecting LoRA: {name}")

                chained_lora_parameters_names.add(f"{name}.lora_A")
                chained_lora_parameters_names.add(f"{name}.lora_B")

    # Ensure only LoRA params are trainable
    # (Double check to be safe)
    for name, param in model.named_parameters():
        if name in chained_lora_parameters_names:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.to(model_dtype)
    model.to(model_device)

    return model


def remove_lora(model: nn.Module, set_requires_grad: bool = False) -> nn.Module:
    """Removes all LoRALinear layers from the model.

    :param model: The pytorch module to remove LoRALinear layers from.
    :param set_requires_grad: Whether to set requires_grad=True for all model parameters
        after removing LoRALinear layers.

    :returns: The input model (removal is in-place).
    """
    modifications = []

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Get parent module.
            parent_name = name.rsplit('.', 1)[0]
            parent = model.get_submodule(parent_name) if '.' in name else model

            # Handle chained LoRA: we only want to unwrap if the parent is NOT a LoRA layer
            # (i.e., we are at the top of the chain).
            if isinstance(parent, LoRALinear):
                continue

            # Find the innermost original layer
            original_layer = module.original_layer
            while isinstance(original_layer, LoRALinear):
                original_layer = original_layer.original_layer

            child_name = name.rsplit('.', 1)[1]
            modifications.append((parent, child_name, original_layer))

    # 2. Apply the modifications
    for parent, child_name, original_layer in modifications:
        setattr(parent, child_name, original_layer)
        print(f"Removed LoRA from: {child_name}")

    if set_requires_grad:
        for param in model.parameters():
            param.requires_grad = True

    return model


def reset_lora_in_place(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.reset()
