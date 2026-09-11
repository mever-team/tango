"""Optional memory-saving gradient-checkpointing variants for Wan's
`CausalWanModel` inference path.

Two independent knobs:

  * **grouped block checkpointing** — `apply_grouped_inference_checkpoint(
    model, group_size)` monkey-patches the model's `_forward_inference`
    to checkpoint *groups* of `group_size` transformer blocks instead of
    one block at a time. NOTE: empirically this INCREASED the per-block
    trajectory backward peak from ~32 GB to ~60 GB on Wan-T2V-1.3B (the
    per-group recompute materialises `group_size` blocks' activations
    simultaneously). Left in for reference / future experimentation but
    not recommended for the TTO per-block trajectory path. Default
    `group_size=1` is a no-op.

  * **intra-block checkpointing** — `enable_intra_block_checkpointing(
    model, enabled)` toggles a flag on every `CausalWanAttentionBlock`
    instance that wraps the self-attention half AND the cross-attn+FFN
    half of `block.forward` in `torch.utils.checkpoint.checkpoint`.
    Saves the per-block FFN intermediate (~84 MB) from being held
    during forward; backward recomputes it. Trade-off: ~2× backward
    time inside each block, but the per-block backward peak drops
    because intermediates live only during recompute. Recommended path
    for fitting the per-block TTO trajectory into a tight memory budget.
    Default is `False` on each block — to behave identically to upstream,
    don't call this helper.

Both helpers are revertible by calling them again with the no-op
argument (`group_size=1` / `enabled=False`). The underlying
modification in `wan/modules/causal_model.py` is gated behind a
default-False instance attribute, so the upstream class is byte-
identical when the flag is left untouched.
"""

from __future__ import annotations

import types
from typing import Any

import torch

from wan.modules.model import sinusoidal_embedding_1d


# Sentinel attribute names attached to the model to keep the monkey-patch
# observable + idempotent.
_PATCH_MARKER_ATTR = "_grouped_inference_checkpoint_patched"
_ORIGINAL_METHOD_ATTR = "_grouped_inference_checkpoint_original"


def apply_grouped_inference_checkpoint(model: torch.nn.Module, group_size: int) -> None:
    """Patch (or unpatch) `model._forward_inference` to checkpoint groups
    of `group_size` transformer blocks.

    `group_size <= 1`: restore the original method if previously patched
    (no-op otherwise). `group_size == 1` is semantically equivalent to no
    grouping; we route to the original method to keep behaviour
    byte-identical to the upstream code.

    `group_size > 1`: install the grouped variant. Re-callable — if a
    different `group_size` is requested later, the prior patch is removed
    first and the new one applied.
    """
    if group_size is None:
        group_size = 1
    group_size = int(group_size)

    already_patched: bool = bool(getattr(model, _PATCH_MARKER_ATTR, False))

    if already_patched:
        # Always restore first, so:
        #   * group_size<=1 leaves the original in place;
        #   * group_size>1 re-installs the grouped variant cleanly even
        #     if the requested group_size differs from the prior one.
        original = getattr(model, _ORIGINAL_METHOD_ATTR)
        model._forward_inference = original
        delattr(model, _ORIGINAL_METHOD_ATTR)
        delattr(model, _PATCH_MARKER_ATTR)

    if group_size <= 1:
        return

    # Snapshot the now-original (possibly already-original-original)
    # bound method so we can restore on unpatch.
    setattr(model, _ORIGINAL_METHOD_ATTR, model._forward_inference)
    setattr(model, _PATCH_MARKER_ATTR, True)
    setattr(model, "gradient_checkpointing_group_size", group_size)

    # The patched method is bound to this specific model instance so
    # `self` resolves correctly (each pipeline owns its own generator
    # model, and we don't want to bleed the patch onto other models that
    # happen to share the class).
    model._forward_inference = types.MethodType(_patched_forward_inference, model)


def enable_intra_block_checkpointing(model: torch.nn.Module, enabled: bool) -> None:
    """Toggle `intra_block_checkpointing` on every transformer block of
    `model`. When `enabled=True`, each block's self-attention and
    cross-attn+FFN halves are wrapped in `torch.utils.checkpoint.
    checkpoint(use_reentrant=False)` (see `wan/modules/causal_model.py`
    `CausalWanAttentionBlock.forward`). When `enabled=False`, the flag
    is cleared and the block's forward takes the upstream byte-identical
    path.

    Idempotent. Safe to call before or after a checkpoint load."""
    enabled = bool(enabled)
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return
    for block in blocks:
        # The attribute is added by `CausalWanAttentionBlock.__init__`,
        # default False. Setting it here is a no-op (writes False to
        # False) when `enabled=False` and the model was constructed
        # from the current source.
        block.intra_block_checkpointing = enabled


def _run_block_group(
    x: torch.Tensor,
    group_blocks: "list[torch.nn.Module]",
    group_block_indices: "list[int]",
    kv_cache: "list[dict]",
    crossattn_cache: "list[dict]",
    current_start: int,
    cache_start: int,
    base_kwargs: "dict[str, Any]",
) -> torch.Tensor:
    """Run `group_blocks` in sequence, indexing into the SHARED per-block
    `kv_cache` / `crossattn_cache` lists by their original block index.
    This is the body of one group's `torch.utils.checkpoint.checkpoint`."""
    for local_idx, block in enumerate(group_blocks):
        block_index = group_block_indices[local_idx]
        kwargs = dict(base_kwargs)
        kwargs["kv_cache"] = kv_cache[block_index]
        kwargs["crossattn_cache"] = crossattn_cache[block_index]
        kwargs["current_start"] = current_start
        kwargs["cache_start"] = cache_start
        x = block(x, **kwargs)
    return x


def _patched_forward_inference(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    kv_cache: "dict | list[dict]" = None,
    crossattn_cache: "dict | list[dict]" = None,
    current_start: int = 0,
    cache_start: int = 0,
):
    """Verbatim copy of `CausalWanModel._forward_inference` with one
    difference: the per-block checkpoint loop is replaced by a grouped
    loop that runs `gradient_checkpointing_group_size` blocks per
    `torch.utils.checkpoint.checkpoint` call. When checkpointing is off
    (e.g. `torch.no_grad()` or `self.gradient_checkpointing == False`),
    the non-checkpointed per-block path takes over (identical to the
    upstream code)."""

    if self.model_type == "i2v":
        assert clip_fea is not None and y is not None

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
    )
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat(x)

    # time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
    )
    e0 = self.time_projection(e).unflatten(
        1, (6, self.dim)
    ).unflatten(dim=0, sizes=t.shape)

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack(
            [
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))]
                )
                for u in context
            ]
        )
    )

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    base_kwargs: dict[str, Any] = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        block_mask=self.block_mask,
    )

    gs = int(getattr(self, "gradient_checkpointing_group_size", 1))
    use_grouped = (
        torch.is_grad_enabled() and self.gradient_checkpointing and gs > 1
    )

    if use_grouped:
        # --- Grouped checkpoint path: one checkpoint() call per group. ---
        for group_start in range(0, len(self.blocks), gs):
            group_end = min(group_start + gs, len(self.blocks))
            group_block_indices = list(range(group_start, group_end))
            group_blocks = [self.blocks[i] for i in group_block_indices]

            x = torch.utils.checkpoint.checkpoint(
                _run_block_group,
                x,
                group_blocks,
                group_block_indices,
                kv_cache,
                crossattn_cache,
                current_start,
                cache_start,
                base_kwargs,
                use_reentrant=False,
            )
    else:
        # --- Non-grouped fallback: identical to the upstream method. ---
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs = dict(base_kwargs)
                kwargs["kv_cache"] = kv_cache[block_index]
                kwargs["current_start"] = current_start
                kwargs["cache_start"] = cache_start
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs = dict(base_kwargs)
                kwargs["kv_cache"] = kv_cache[block_index]
                kwargs["crossattn_cache"] = crossattn_cache[block_index]
                kwargs["current_start"] = current_start
                kwargs["cache_start"] = cache_start
                x = block(x, **kwargs)

    # head
    x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return torch.stack(x)
