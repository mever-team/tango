"""Memory-optimised variant of `causal_inference_tto.py`.

Two changes vs. the reference implementation in `causal_inference_tto.py`,
both designed to leave every observable output within the GPU noise floor:

1. **In-place critic pre-compute.** The per-(block, sub-timestep) critic
   pre-compute (`critic_denoised_pred`) and the `measure_only` forward run
   on the *live* `state.kv_cache1` and `state.crossattn_cache` (no cache
   clone). Before the forward we snapshot just the regions the model will
   mutate (see `_save_kv_cache_for_inplace`); after the forward we paste
   them back, leaving the live cache byte-identical to its pre-forward
   state. Two branches handle the two cache states:

     * **no-roll**: when the cache has room or the model's internal
       predicate is otherwise False, we save the slice
       `cache[local_start_post : local_end_post]` per layer (~864 MB).
     * **rolling**: when the model's internal shift-then-write fires
       (`wan/modules/causal_model.py:206-222`), we save only the evicted
       region `cache[sink : sink + num_evicted]` per layer (~864 MB); the
       chunked right-to-left in-place copy in `_restore_kv_cache_for_inplace`
       undoes the left-shift and reconstructs the rest of the cache from
       its post-forward state.

   These sites are wrapped in `torch.no_grad()` so there is no autograd
   graph that depends on the cache state. Eliminates ~5.85 GB versus a
   full clone in both branches.

   The inner `_forward_backward` path (trainable + critic look-ahead +
   backward) still clones the cache: gradient checkpointing later re-runs
   those forward passes during backward, and if the cache had been mutated
   between the original forward and the recomputation it would observe
   different K/V state.

2. **Skip `_kv_cache_batch_repeat` at B=1.** The reference implementation
   always calls `tensor.repeat((B, 1, 1, 1))` to materialise the batched
   cache, which at B=1 is a copy of the (large) cache for no reason. We
   skip the call entirely when B == 1 and pass the original cache through.
   Saves ~5.85 GB on the canonical config.

Why no LoRA-toggle critic in this revision:
    A pure runtime `disable_lora` flag is incompatible with gradient
    checkpointing — the `with lora.disabled(model):` block has already
    exited by the time the autograd engine replays the forward during
    backward, so the replay sees a different number/type/shape of saved
    tensors. The `tto/lora.py` `disabled(...)` context manager is still
    safe in `torch.no_grad()` paths but we don't actually need it here:
    the existing `tto_trainer.critic_model` is used for every critic call
    (matching the reference), and we leave dropping that copy to a future
    revision that integrates the LoRA toggle into the model's forward
    signature so the replay sees a consistent state.

Behavioural invariants preserved:
  * Same number of forwards per (block, sub-timestep, epoch).
  * Same sequence of writes to the cache.
  * Same RNG consumption order (no extra `torch.randn` calls, no skipped ones).
  * Same scheduler arithmetic.
  * Same gradient pathway: the trainable forward writes to the live cache,
    the critic look-ahead reads from it, the loss flows back through both.

Not yet supported here:
  * `tto.batch_size > 1`: at B=1 the cache `batch_repeat` is a no-op
    semantically but the original implementation still allocates a copy. We
    skip the allocation at B=1. For B>1 the optimisation falls back to the
    reference path (allocating the batched cache). See `_compute_loss` below.
"""

import math
import time
from types import SimpleNamespace
from typing import Any, List, Optional

import torch

from tto import trainer
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalInferenceTTOOptimizedPipeline(torch.nn.Module):
    """Drop-in replacement for `CausalInferenceTTOPipeline` with the
    memory optimisations described in the module docstring."""

    NUM_TRANSFORMER_BLOCKS: int = 30
    FRAME_SEQ_LENGTH: int = 1560

    def __init__(
        self,
        args,
        device,
        generator=None,
        text_encoder=None,
        vae=None,
    ):
        super().__init__()
        self.args = args
        self.device = device

        self.generator = (
            WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
            if generator is None else generator
        )
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_frame_per_block: int = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size: int = self.generator.model.local_attn_size

        self.kv_cache1: Optional[list[dict[str, torch.Tensor]]] = None
        self.crossattn_cache: Optional[list[dict[str, Any]]] = None

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        # Optional: coarser gradient checkpointing for the inference
        # forward (idea-4 per-block trajectory chains four autograd
        # forwards; grouped checkpointing saves ~1.4 GB of stored block-
        # boundary activations per trajectory at group_size=5). Set via
        # `tto.gradient_checkpointing_group_size` (default 1 = upstream
        # per-block behaviour). The helper monkey-patches THIS model
        # instance only; reverting is `apply_grouped_inference_checkpoint(
        # model, 1)` or simply not setting the config key.
        gs = int(getattr(args, "tto", {}).get("gradient_checkpointing_group_size", 1))
        if gs > 1:
            from tto.grouped_checkpoint import apply_grouped_inference_checkpoint
            apply_grouped_inference_checkpoint(self.generator.model, gs)
            print(f"[opt] Grouped checkpointing enabled (group_size={gs}).")

        # Optional: intra-block gradient checkpointing. Wraps the self-
        # attention and cross-attn+FFN halves of every transformer block in
        # `torch.utils.checkpoint.checkpoint(use_reentrant=False)`, so the
        # FFN's `[B, T, ffn_dim]` intermediate (~84 MB / block for Wan-
        # 1.3B) is recomputed during backward instead of stored. Toggle
        # via `tto.intra_block_gradient_checkpointing` (default False =
        # upstream byte-identical path through `CausalWanAttentionBlock.
        # forward`). Reverting is `enable_intra_block_checkpointing(
        # model, False)` or simply not setting the config key.
        intra_ckpt = bool(getattr(args, "tto", {}).get("intra_block_gradient_checkpointing", False))
        if intra_ckpt:
            from tto.grouped_checkpoint import enable_intra_block_checkpointing
            enable_intra_block_checkpointing(self.generator.model, True)
            print("[opt] Intra-block gradient checkpointing enabled.")

        # Memory profiling state.
        self._profile_memory: bool = False

        print(f"[opt] KV inference with {self.num_frame_per_block} frames per block.")

    # ------------------------------------------------------------------
    # Helpers: KV-cache index save/restore + critic toggle.
    # ------------------------------------------------------------------

    def _save_kv_cache_window(
        self,
        kv_cache: list[dict[str, torch.Tensor]],
        n_frames_written: int,
    ) -> dict:
        """Capture exactly what an in-place forward will mutate:
            * `local_end_index` / `global_end_index` (a few `long` scalars).
            * The slice of K/V values at `[local_end_index : local_end_index
              + n_frames_written * FRAME_SEQ_LENGTH]` per layer — i.e. the
              window the forward is about to write into.

        Restoring this snapshot afterwards makes the live cache observably
        identical to its pre-forward state, which is what the reference
        implementation achieves via `_clone_cache` (at full ~5.85 GB cost).
        Per-layer slice cost is `n_frames * 1560 * 12 * 128 * 2 (K+V) * 2 bytes
        ≈ 14.6 MB * n_frames`. For a single block of 3 frames × 30 layers,
        total is ≈ 1.3 GB — versus a full clone at ≈ 5.85 GB."""
        tokens_per_frame = self.FRAME_SEQ_LENGTH
        slice_tokens = n_frames_written * tokens_per_frame
        per_layer = []
        for entry in kv_cache:
            local_end = int(entry["local_end_index"][0].item())
            cache_size = entry["k"].size(1)
            slice_end = min(local_end + slice_tokens, cache_size)
            per_layer.append({
                "local_end_index": entry["local_end_index"].detach().clone(),
                "global_end_index": entry["global_end_index"].detach().clone(),
                "k_slice": entry["k"][:, local_end:slice_end].detach().clone(),
                "v_slice": entry["v"][:, local_end:slice_end].detach().clone(),
                "slice_start": local_end,
                "slice_end": slice_end,
            })
        return {"layers": per_layer}

    @staticmethod
    def _restore_kv_cache_window(
        kv_cache: list[dict[str, torch.Tensor]],
        saved: dict,
    ) -> None:
        for entry, layer_snap in zip(kv_cache, saved["layers"]):
            entry["local_end_index"].copy_(layer_snap["local_end_index"])
            entry["global_end_index"].copy_(layer_snap["global_end_index"])
            s, e = layer_snap["slice_start"], layer_snap["slice_end"]
            entry["k"][:, s:e].copy_(layer_snap["k_slice"])
            entry["v"][:, s:e].copy_(layer_snap["v_slice"])

    def _snapshot_kv_at_last_block(
        self,
        kv_cache: list[dict[str, torch.Tensor]],
        current_num_frames: int,
    ) -> dict:
        """Snapshot the K/V slice at the MOST RECENTLY WRITTEN block, i.e.
        positions `[local_end_index - current_num_frames * FRAME_SEQ_LENGTH :
        local_end_index)` per layer.

        Used by `lookahead_target=dual`: after the trainable forward has
        written its K/V at the current block range, we run a within-block
        critic look-ahead that overwrites that same slice in-place; we then
        restore the snapshot so the next-block critic look-ahead reads the
        trainable's (not the within-block-look-ahead's) K/V.

        Cost: `~current_num_frames * FRAME_SEQ_LENGTH * 12 * 128 * 2 (K+V) * 30 layers * 2 bytes`
        ≈ 85 MB for a 3-frame block on Wan-1.3B. Complement to
        `_save_kv_cache_window`, which snapshots the slice BEFORE the
        first in-place forward writes there.
        """
        tokens = current_num_frames * self.FRAME_SEQ_LENGTH
        per_layer: list[dict] = []
        for entry in kv_cache:
            local_end = int(entry["local_end_index"][0].item())
            slice_start = max(0, local_end - tokens)
            per_layer.append({
                "local_end_index": entry["local_end_index"].detach().clone(),
                "global_end_index": entry["global_end_index"].detach().clone(),
                "k_slice": entry["k"][:, slice_start:local_end].detach().clone(),
                "v_slice": entry["v"][:, slice_start:local_end].detach().clone(),
                "slice_start": slice_start,
                "slice_end": local_end,
            })
        return {"layers": per_layer}

    @staticmethod
    def _restore_kv_at_last_block(
        kv_cache: list[dict[str, torch.Tensor]],
        saved: dict,
    ) -> None:
        """Restore the snapshot taken by `_snapshot_kv_at_last_block`."""
        for entry, layer_snap in zip(kv_cache, saved["layers"]):
            entry["local_end_index"].copy_(layer_snap["local_end_index"])
            entry["global_end_index"].copy_(layer_snap["global_end_index"])
            s, e = layer_snap["slice_start"], layer_snap["slice_end"]
            entry["k"][:, s:e].copy_(layer_snap["k_slice"])
            entry["v"][:, s:e].copy_(layer_snap["v_slice"])

    @staticmethod
    def _save_crossattn_is_init(
        crossattn_cache: list[dict[str, Any]],
    ) -> list[bool]:
        return [bool(entry["is_init"]) for entry in crossattn_cache]

    @staticmethod
    def _restore_crossattn_is_init(
        crossattn_cache: list[dict[str, Any]],
        saved: list[bool],
    ) -> None:
        # We intentionally leave the prompt K/V tensors in place even if
        # `is_init` was False going in. They're a pure function of the prompt
        # and matches whatever a fresh forward would compute. Restoring the
        # flag would just force the next forward to recompute the same values.
        # We DO restore the flag if it was True going in but somehow flipped —
        # an over-defensive belt-and-braces.
        for entry, prev in zip(crossattn_cache, saved):
            entry["is_init"] = entry["is_init"] or prev

    def _would_overflow_after_writes(
        self,
        kv_cache: list[dict[str, torch.Tensor]],
        n_blocks_to_write: int,
    ) -> bool:
        """Coarse predicate: would `n_blocks_to_write` blocks of writes push
        past the end of the local window? Used as a fast gate before the
        per-layer / per-step check in `_save_kv_cache_for_inplace`.

        NOTE: this is looser than the model's actual rolling predicate (see
        `_model_will_roll`); kept only for `_measure_loss`'s clone fallback,
        which has multiple back-to-back forwards whose individual roll
        behaviour is harder to summarise."""
        cache_size = kv_cache[0]["k"].size(1)
        cache_end = int(kv_cache[0]["local_end_index"][0].item())
        tokens_per_block = self.num_frame_per_block * self.FRAME_SEQ_LENGTH
        return cache_end + n_blocks_to_write * tokens_per_block > cache_size

    def _sink_tokens(self) -> int:
        """Sink-attention region size in tokens. The model keeps the first
        `sink_size` frames pinned at positions
        `[0 : sink_size * FRAME_SEQ_LENGTH]` and never shifts them.
        Default is 0 (no sink) for Wan2.1-T2V-1.3B."""
        sink_size = int(getattr(self.generator.model, "sink_size", 0) or 0)
        return sink_size * self.FRAME_SEQ_LENGTH

    @staticmethod
    def _set_cache_grad_safe_reads(wrapper, flag: bool) -> None:
        """Toggle clone-on-read of the attended KV-cache window (see
        `CausalWanSelfAttention.cache_grad_safe_reads` in
        `wan/modules/causal_model.py`) on every block of a
        `WanDiffusionWrapper`'s model. Part of the lookahead
        cache-gradient fix; default state is False everywhere."""
        for block in wrapper.model.blocks:
            block.self_attn.cache_grad_safe_reads = flag

    def _lookahead_fix_restore(self, tto_trainer, fix_wrappers: list) -> None:
        """Undo the per-epoch state of the lookahead cache-gradient fix:
        clear the clone-on-read flags and re-enable gradient
        checkpointing when the config asks for it. Called before every
        return of `_compute_loss` when the fix was engaged."""
        if not fix_wrappers:
            return
        reenable = bool(tto_trainer.config.tto.get("gradient_checkpointing", False))
        for w in fix_wrappers:
            self._set_cache_grad_safe_reads(w, False)
            if reenable:
                w.enable_gradient_checkpointing()

    def _sink_ema_enabled(self) -> bool:
        """Whether the model's sink-EMA (Reward-Forcing's `compression_alpha`)
        is active. When True, an in-place forward that triggers cache
        rolling ALSO modifies the sink positions
        `[0 : sink_size * FRAME_SEQ_LENGTH]` via
        `sink ← alpha * sink + (1 - alpha) * evicted`. In that case
        `_save_kv_cache_for_inplace` must also snapshot the sink so
        `_restore_kv_cache_for_inplace` can undo the blend."""
        block0 = self.generator.model.blocks[0]
        alpha = getattr(block0.self_attn, "compression_alpha", None)
        return alpha is not None and self._sink_tokens() > 0

    def _model_will_roll(
        self,
        kv_cache: list[dict[str, torch.Tensor]],
        current_start_frame: int,
        num_new_frames: int,
    ) -> bool:
        """Predict whether an upcoming forward at `current_start_frame *
        FRAME_SEQ_LENGTH` writing `num_new_frames * FRAME_SEQ_LENGTH` new
        tokens will trigger the model's internal shift-then-write rolling
        (see `wan/modules/causal_model.py:206-222`). Boolean is byte-for-byte
        the same as the model's own predicate."""
        if self.local_attn_size == -1:
            return False
        entry = kv_cache[0]
        cache_size = entry["k"].size(1)
        local_end = int(entry["local_end_index"][0].item())
        global_end = int(entry["global_end_index"][0].item())
        num_new = num_new_frames * self.FRAME_SEQ_LENGTH
        current_start = current_start_frame * self.FRAME_SEQ_LENGTH
        current_end = current_start + num_new
        return current_end > global_end and num_new + local_end > cache_size

    def _save_kv_cache_for_inplace(
        self,
        kv_cache: list[dict[str, torch.Tensor]],
        current_start_frame: int,
        num_new_frames: int,
    ) -> dict:
        """Snapshot exactly enough state to undo an in-place model forward.

        Save buffer is ~num_new_frames worth of tokens per layer (~864 MB
        total on the canonical config) regardless of whether the model's
        internal shift-then-write fires, versus ~5.85 GB for a full clone.

        Two branches per layer:
          * **rolling** (`_model_will_roll` is True): the model overwrites
            cache[sink : sink + num_evicted] with a shifted copy of
            cache[sink + num_evicted : sink + num_evicted + num_rolled] and
            writes new K/V at cache[local_start_post : local_end_post]. We
            save the evicted region only — the un-shift in restore
            reconstructs everything to the right from the post-forward state.
          * **no_roll**: the model leaves the cache layout intact and only
            writes at cache[local_start_post : local_end_post]. We save just
            that slice."""
        sink = self._sink_tokens()
        sink_ema = self._sink_ema_enabled()
        num_new = num_new_frames * self.FRAME_SEQ_LENGTH
        current_start = current_start_frame * self.FRAME_SEQ_LENGTH
        per_layer: list[dict] = []
        for entry in kv_cache:
            cache_size = entry["k"].size(1)
            local_end = int(entry["local_end_index"][0].item())
            global_end = int(entry["global_end_index"][0].item())
            current_end = current_start + num_new
            will_roll = (
                self.local_attn_size != -1
                and current_end > global_end
                and num_new + local_end > cache_size
            )
            if will_roll:
                num_evicted = num_new + local_end - cache_size
                num_rolled = local_end - num_evicted - sink
                layer_snap: dict = {
                    "strategy": "rolling",
                    "local_end_index": entry["local_end_index"].detach().clone(),
                    "global_end_index": entry["global_end_index"].detach().clone(),
                    "saved_k": entry["k"][:, sink:sink + num_evicted].detach().clone(),
                    "saved_v": entry["v"][:, sink:sink + num_evicted].detach().clone(),
                    "sink": sink,
                    "num_evicted": num_evicted,
                    "num_rolled": num_rolled,
                }
                # If the model's sink-EMA (`compression_alpha`) is on, the
                # rolling forward also blends the sink positions —
                # `sink ← alpha * sink + (1 - alpha) * evicted` — so we
                # need to snapshot the pre-blend sink to undo it. Cost is
                # `sink_size * FRAME_SEQ_LENGTH * K+V * bf16` per layer
                # (~28 MB per layer at sink_size=3, ~840 MB across the
                # 30-layer model) — held transiently during the critic
                # precompute forward, released on restore.
                if sink > 0 and sink_ema:
                    layer_snap["saved_sink_k"] = entry["k"][:, :sink].detach().clone()
                    layer_snap["saved_sink_v"] = entry["v"][:, :sink].detach().clone()
                per_layer.append(layer_snap)
            else:
                local_end_post = local_end + max(0, current_end - global_end)
                local_start_post = local_end_post - num_new
                per_layer.append({
                    "strategy": "no_roll",
                    "local_end_index": entry["local_end_index"].detach().clone(),
                    "global_end_index": entry["global_end_index"].detach().clone(),
                    "saved_k": entry["k"][:, local_start_post:local_end_post].detach().clone(),
                    "saved_v": entry["v"][:, local_start_post:local_end_post].detach().clone(),
                    "local_start_post": local_start_post,
                    "local_end_post": local_end_post,
                })
        return {"layers": per_layer}

    @staticmethod
    def _restore_kv_cache_for_inplace(
        kv_cache: list[dict[str, torch.Tensor]],
        saved: dict,
    ) -> None:
        """Reverse an in-place forward snapshotted by
        `_save_kv_cache_for_inplace`.

        For the **rolling** branch:
          1. Chunked right-to-left in-place copy
             `cache[sink + num_evicted : sink + num_evicted + num_rolled]
              <- cache[sink : sink + num_rolled]`,
             undoing the model's left-shift. Chunks are exactly `num_evicted`
             tokens wide so each chunk's source/destination are guaranteed
             non-overlapping (`dst_start == src_end`); we process them
             right-to-left so the (i-1)-th chunk's source is never clobbered
             by the i-th chunk's write (the i-th chunk only writes to
             positions >= i*chunk + num_evicted, all to the right of chunk
             i-1's read region).
          2. Paste the saved evicted region back at cache[sink : sink + num_evicted].

        For **no_roll** we just restore the saved write slice in place.
        Indices are always restored last."""
        for entry, snap in zip(kv_cache, saved["layers"]):
            if snap["strategy"] == "rolling":
                sink = snap["sink"]
                num_evicted = snap["num_evicted"]
                num_rolled = snap["num_rolled"]
                # Step 1: chunked in-place shift right (undoes the model's shift).
                chunk_size = num_evicted
                n_chunks = (num_rolled + chunk_size - 1) // chunk_size
                for i in range(n_chunks - 1, -1, -1):
                    src_start = sink + i * chunk_size
                    src_end = min(src_start + chunk_size, sink + num_rolled)
                    dst_start = src_start + num_evicted
                    dst_end = src_end + num_evicted
                    entry["k"][:, dst_start:dst_end] = entry["k"][:, src_start:src_end]
                    entry["v"][:, dst_start:dst_end] = entry["v"][:, src_start:src_end]
                # Step 2: restore the evicted region.
                entry["k"][:, sink:sink + num_evicted].copy_(snap["saved_k"])
                entry["v"][:, sink:sink + num_evicted].copy_(snap["saved_v"])
                # Step 3 (only under sink-EMA): restore the sink positions
                # so the EMA blend the model applied during the forward is
                # undone. Otherwise (plain-pin sink) the sink positions
                # were never touched by the forward and we leave them alone.
                if "saved_sink_k" in snap:
                    entry["k"][:, :sink].copy_(snap["saved_sink_k"])
                    entry["v"][:, :sink].copy_(snap["saved_sink_v"])
            else:
                ls = snap["local_start_post"]
                le = snap["local_end_post"]
                entry["k"][:, ls:le].copy_(snap["saved_k"])
                entry["v"][:, ls:le].copy_(snap["saved_v"])
            entry["local_end_index"].copy_(snap["local_end_index"])
            entry["global_end_index"].copy_(snap["global_end_index"])

    def _clone_kv_cache(
        self,
        kv_cache: list[dict[str, torch.Tensor]],
        device: torch.device,
    ) -> list[dict[str, torch.Tensor]]:
        return [
            {
                k: (v.detach().clone().to(device).requires_grad_(False)
                    if isinstance(v, torch.Tensor) else v)
                for k, v in entry.items()
            }
            for entry in kv_cache
        ]

    def _clone_crossattn_cache(
        self,
        crossattn_cache: list[dict[str, Any]],
        device: torch.device,
    ) -> list[dict[str, Any]]:
        return [
            {
                k: (v.detach().clone().to(device).requires_grad_(False)
                    if isinstance(v, torch.Tensor) else v)
                for k, v in entry.items()
            }
            for entry in crossattn_cache
        ]

    @staticmethod
    def _crossattn_cache_batch_expand(
        cache: list[dict[str, Any]],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        for entry in cache:
            entry["k"] = entry["k"].expand((batch_size, -1, -1, -1))
            entry["v"] = entry["v"].expand((batch_size, -1, -1, -1))
        return cache

    @staticmethod
    def _kv_cache_batch_repeat(
        cache: list[dict[str, torch.Tensor]],
        batch_size: int,
    ) -> list[dict[str, torch.Tensor]]:
        for entry in cache:
            entry["k"] = entry["k"].repeat((batch_size, 1, 1, 1))
            entry["v"] = entry["v"].repeat((batch_size, 1, 1, 1))
            entry["global_end_index"] = entry["global_end_index"].repeat(batch_size)
            entry["local_end_index"] = entry["local_end_index"].repeat(batch_size)
        return cache

    def _mem(self, tag: str) -> None:
        if not self._profile_memory:
            return
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        max_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"[mem-opt] {tag:<60}  alloc={alloc:6.2f}  peak={max_alloc:6.2f}  reserved={reserved:6.2f} GB", flush=True)

    def _viz_should_capture(self, block_index: int, subtimestep_index: int) -> bool:
        """Predicate: should we record a viz latent for (block, sub-timestep)?
        Honours `tto.viz_blocks` / `tto.viz_subtimesteps`. When viz is off
        (`tto.viz_trajectory=false`), always returns False."""
        if not self._viz_enabled:
            return False
        if self._viz_blocks is not None and block_index not in self._viz_blocks:
            return False
        if (
            self._viz_subtimesteps is not None
            and subtimestep_index not in self._viz_subtimesteps
        ):
            return False
        return True

    def _viz_record(
        self,
        latent: torch.Tensor,
        block_index: int,
        subtimestep_index: int,
        epoch: "int | str",
    ) -> None:
        """Detach and copy a `denoised_pred` latent to CPU bf16, appending
        to `self._viz_buffer`. Called from `_compute_loss` (per epoch) and
        from `generate()` (post-TTO actual-denoising forward, with
        `epoch="final"`). No-op when viz is disabled."""
        if self._viz_buffer is None:
            return
        if not self._viz_should_capture(block_index, subtimestep_index):
            return
        self._viz_buffer.append({
            "block": block_index,
            "subtimestep": subtimestep_index,
            "epoch": epoch,
            # Keep bf16 on CPU to halve memory vs fp32. Conversion back to
            # the model dtype happens at decode time.
            "latent": latent.detach().to("cpu", torch.bfloat16, copy=True),
        })

    @torch.enable_grad()
    def generate(
        self,
        tto_trainer: trainer.Trainer,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_rgb: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, Any]]:
        """Run TTO-enabled generation.

        Modes:
            * V2V / I2V: pass `initial_rgb` (shape `B x C x T x H x W`). The
              first frames of the output are the VAE-encoded conditioning;
              the remaining `noise.shape[1]` frames are denoised.
            * T2V: pass `initial_rgb=None`. The full `noise.shape[1]` frames
              are denoised from a freshly-initialised KV cache.
        """
        device: torch.device = noise.device
        self._profile_memory = bool(tto_trainer.config.tto.get("profile_memory", False))
        if self._profile_memory:
            torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t_start: float = time.perf_counter()
        self._mem("generate() entry")

        # ---- Optional: capture per-epoch x0 predictions for visualisation. ----
        # When `tto.viz_trajectory=true`, accumulate the trainable model's
        # `denoised_pred` at the START of each TTO epoch (= LoRA state BEFORE
        # that epoch's update; so "epoch 0" is the unoptimised model) and,
        # if `tto.viz_include_final=true` (default), also the post-TTO
        # actual-denoising output. Latents are kept on CPU as bf16 (~600 KB
        # per (block, sub-timestep, epoch)) and batch-decoded at the end of
        # generate() into RGB MP4s the entry script writes alongside the
        # main video.
        self._viz_enabled: bool = bool(
            tto_trainer.config.tto.get("viz_trajectory", False)
        )
        viz_blocks_cfg = tto_trainer.config.tto.get("viz_blocks", None)
        self._viz_blocks: Optional[set[int]] = (
            set(int(b) for b in viz_blocks_cfg) if viz_blocks_cfg is not None else None
        )
        viz_subtimesteps_cfg = tto_trainer.config.tto.get("viz_subtimesteps", None)
        if viz_subtimesteps_cfg is None:
            # Default: only the last sub-timestep (the most refined x0).
            self._viz_subtimesteps: set[int] = {len(self.denoising_step_list) - 1}
        else:
            self._viz_subtimesteps = set(int(s) for s in viz_subtimesteps_cfg)
        self._viz_include_final: bool = bool(
            tto_trainer.config.tto.get("viz_include_final", True)
        )
        # Burn a "block NN | subtimestep S | epoch E" label into the
        # top-left corner of every decoded viz frame (default on).
        # Disable with `tto.viz_annotate=false` for clean frames.
        self._viz_annotate: bool = bool(
            tto_trainer.config.tto.get("viz_annotate", True)
        )
        self._viz_font = None  # lazily built by `_viz_annotate_frames`
        # Number of preceding latent frames from the committed rollout to
        # prepend as decode context for each viz chapter. The Wan VAE
        # decoder is temporally causal — decoding a 3-latent snippet in
        # isolation reconstructs its first frames poorly (no preceding
        # features to condition on). Prefixing the block's actual
        # neighbours and dropping their decoded frames gives every
        # chapter the same decode quality as the merged video. 0 turns
        # the prefixing off.
        self._viz_decode_context_frames: int = int(
            tto_trainer.config.tto.get("viz_decode_context_frames", 3)
        )
        # Diff overlays: for every chapter, also emit a clip highlighting
        # the ABSOLUTE per-pixel difference against the epoch-0
        # (unoptimised) reference chapter as a single-colour (red)
        # overlay whose opacity ramps with `|delta| / viz_diff_scale`.
        # The scale is FIXED (in [0, 1] pixel units, default 0.10) so
        # clips are comparable across chapters/blocks/sub-timesteps; the
        # dimmed grayscale of the reference shows through wherever
        # nothing changed.
        self._viz_diff_overlay: bool = bool(
            tto_trainer.config.tto.get("viz_diff_overlay", True)
        )
        self._viz_diff_scale: float = float(
            tto_trainer.config.tto.get("viz_diff_scale", 0.10)
        )
        self._viz_ssim_window: Optional[torch.Tensor] = None
        # Buffer of {block, subtimestep, epoch, latent_cpu_bf16}. `epoch` is
        # an int 0..B-1 for per-epoch captures and the string "final" for the
        # post-TTO actual-denoising forward.
        self._viz_buffer: Optional[list[dict[str, Any]]] = (
            [] if self._viz_enabled else None
        )

        # ---- Encode the conditioning video (V2V/I2V only). ----
        torch.cuda.synchronize(device)
        t_vae_enc_start: float = time.perf_counter()
        initial_latent: Optional[torch.Tensor] = None
        if initial_rgb is not None:
            with torch.no_grad():
                initial_latent = self.vae(initial_rgb, mode="encode").to(
                    device, dtype=noise.dtype
                )
        torch.cuda.synchronize(device)
        vae_encode_seconds: float = time.perf_counter() - t_vae_enc_start
        self._mem("after VAE encode of initial_rgb")

        batch_size, num_frames, num_channels, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks: int = num_frames // self.num_frame_per_block
        # T2V has no conditioning frames, so `num_input_frames = 0` and the
        # absolute frame index in `output` equals the index in `noise`.
        num_input_frames: int = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames: int = num_frames + num_input_frames

        # ---- Encode the prompts and offload the text encoder. ----
        torch.cuda.synchronize(device)
        t_text_start: float = time.perf_counter()
        with torch.no_grad():
            self.text_encoder.to(device)
            conditional_dict: dict[str, torch.Tensor] = self.text_encoder(text_prompts=text_prompts)
            self.text_encoder.to("cpu")
        torch.cuda.synchronize(device)
        text_encode_seconds: float = time.perf_counter() - t_text_start
        self._mem("after text encode + offload")

        # ---- Initialize the KV caches and (V2V/I2V only) seed with the conditioning. ----
        with torch.no_grad():
            self._initialize_kv_cache(batch_size, noise.dtype, device)
            self._initialize_crossattn_cache(batch_size, noise.dtype, device)
            self._mem("after KV cache init")

            output: torch.Tensor = torch.zeros(
                [batch_size, num_output_frames, num_channels, height, width],
                device=device,
                dtype=noise.dtype,
            )
            current_start_frame: int = 0
            if initial_latent is not None:  # V2V generation
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks: int = num_input_frames // self.num_frame_per_block
                zero_timestep: torch.Tensor = torch.zeros(
                    [batch_size, 1], device=device, dtype=torch.int64
                )
                for _ in range(num_input_blocks):
                    ref: torch.Tensor = initial_latent[
                        :, current_start_frame:current_start_frame + self.num_frame_per_block
                    ]
                    output[
                        :, current_start_frame:current_start_frame + self.num_frame_per_block
                    ] = ref
                    tto_trainer.model(
                        noisy_image_or_video=ref,
                        conditional_dict=conditional_dict,
                        timestep=zero_timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.FRAME_SEQ_LENGTH,
                    )
                    current_start_frame += self.num_frame_per_block

        self._mem("after seeding KV cache with conditioning frames")

        # ---- Per-block, per-sub-timestep TTO + actual denoising. ----
        torch.cuda.synchronize(device)
        t_rollout_start: float = time.perf_counter()
        per_step_losses: dict[int, dict[int, dict[int, dict[str, float]]]] = {}

        for block_index in range(num_blocks):
            self._mem(f"block {block_index:>2} start")
            block_losses: dict[int, dict[int, dict[str, float]]] = {}
            per_step_losses[block_index] = block_losses
            current_num_frames: int = self.num_frame_per_block

            noisy_input: torch.Tensor = noise[
                :,
                current_start_frame - num_input_frames:
                current_start_frame + current_num_frames - num_input_frames,
            ]

            # `tto.optimization_scope` picks between historical
            # per-sub-timestep TTO and idea 4's per-block trajectory TTO.
            optimization_scope: str = str(
                tto_trainer.config.tto.get("optimization_scope", "per_subtimestep")
            )

            block_traj_state: Optional[SimpleNamespace] = None
            if optimization_scope == "per_block":
                block_traj_state = SimpleNamespace(
                    initial_latent=initial_latent,
                    current_start_frame=current_start_frame,
                    noisy_input=noisy_input,
                    kv_cache1=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                )
                self._mem(f"  block {block_index:>2} per-block TTO pre")
                block_losses["block"] = self._optimize_block_trajectory(
                    tto_trainer=tto_trainer,
                    state=block_traj_state,
                    conditional_dict=conditional_dict,
                    block_index=block_index,
                    batch_size=batch_size,
                    current_num_frames=current_num_frames,
                    noise=noise,
                    num_input_frames=num_input_frames,
                )
                self._mem(f"  block {block_index:>2} per-block TTO post")

            denoised_pred: Optional[torch.Tensor] = None
            for index in range(len(self.denoising_step_list)):
                current_timestep: float = float(self.denoising_step_list[index])
                timestep: torch.Tensor = torch.full(
                    [batch_size, current_num_frames],
                    current_timestep,
                    device=device,
                    dtype=torch.float32,
                )

                state = SimpleNamespace(
                    initial_latent=initial_latent,
                    current_start_frame=current_start_frame,
                    noisy_input=noisy_input,
                    kv_cache1=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                )

                if optimization_scope == "per_subtimestep":
                    self._mem(f"  block {block_index:>2} step {index} pre-optimize")
                    block_losses[index] = self._optimize_block_step(
                        tto_trainer=tto_trainer,
                        state=state,
                        timestep=timestep,
                        conditional_dict=conditional_dict,
                        index=index,
                        block_index=block_index,
                        batch_size=batch_size,
                        current_num_frames=current_num_frames,
                        noise=noise,
                        num_input_frames=num_input_frames,
                    )
                    self._mem(f"  block {block_index:>2} step {index} post-optimize")

                # --- Run the actual denoising step with the optimized model. ---
                with torch.no_grad():
                    _, denoised_pred = tto_trainer.model(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.FRAME_SEQ_LENGTH,
                    )
                    # Viz capture — the fully-optimised x0 for this
                    # (block, sub-timestep). Tag with epoch="final" so the
                    # entry script can place it at the end of the per-block
                    # MP4 timeline. Only fires when viz_include_final=true.
                    if self._viz_include_final:
                        self._viz_record(denoised_pred, block_index, index, "final")
                    if index < len(self.denoising_step_list) - 1:
                        next_timestep: float = float(self.denoising_step_list[index + 1])
                        # `tto.reuse_critic_noise=true` source depends on scope:
                        #   per_subtimestep: `state.last_critic_noise`
                        #   per_block:       `block_traj_state.last_trajectory_noises[index]`
                        reuse_for_gen: bool = bool(
                            tto_trainer.config.tto.get("reuse_critic_noise", False)
                        )
                        next_noise: Optional[torch.Tensor] = None
                        if reuse_for_gen:
                            if optimization_scope == "per_block":
                                tn = (
                                    block_traj_state.last_trajectory_noises
                                    if block_traj_state is not None else None
                                )
                                if tn is not None and index < len(tn):
                                    next_noise = tn[index]
                            else:
                                last_noise = getattr(state, "last_critic_noise", None)
                                if last_noise is not None:
                                    next_noise = last_noise
                        if next_noise is None:
                            next_noise = torch.randn_like(denoised_pred)
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.detach().flatten(0, 1),
                            next_noise.flatten(0, 1),
                            torch.full(
                                [batch_size * current_num_frames],
                                next_timestep,
                                device=device,
                                dtype=torch.float32,
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])

            # --- Persist the optimized output of the block. ---
            assert denoised_pred is not None
            with torch.no_grad():
                output[
                    :, current_start_frame:current_start_frame + current_num_frames
                ] = denoised_pred

                context_timestep: torch.Tensor = torch.full_like(
                    timestep, self.args.context_noise
                )
                tto_trainer.model(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.FRAME_SEQ_LENGTH,
                )

            current_start_frame += current_num_frames
            self._mem(f"block {block_index:>2} end")

        torch.cuda.synchronize(device)
        rollout_seconds: float = time.perf_counter() - t_rollout_start
        self._mem("after rollout")

        self.kv_cache1 = None
        self.crossattn_cache = None

        torch.cuda.synchronize(device)
        t_dec_start: float = time.perf_counter()
        with torch.no_grad():
            video: torch.Tensor = self.vae(output.to(device), mode="decode", use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1).float().cpu()
        torch.cuda.synchronize(device)
        vae_decode_seconds: float = time.perf_counter() - t_dec_start

        # ---- Optional: VAE-decode the captured per-epoch x0 latents. ----
        # Grouped by (block, sub-timestep); within each group, latents are
        # concatenated along the temporal axis in epoch order
        # `[0, 1, ..., B-1, "final"]` so the resulting tensor reads as a
        # single "evolution" clip the entry script can write to one MP4.
        viz_outputs: Optional[dict[str, torch.Tensor]] = None
        viz_decode_seconds: float = 0.0
        if self._viz_buffer is not None and len(self._viz_buffer) > 0:
            torch.cuda.synchronize(device)
            t_viz_start: float = time.perf_counter()
            viz_outputs = self._decode_viz_buffer(
                self._viz_buffer, device, noise.dtype,
                output_latents=output, num_input_frames=num_input_frames,
            )
            torch.cuda.synchronize(device)
            viz_decode_seconds = time.perf_counter() - t_viz_start
            # Free the latents buffer; the decoded uint8 tensors in
            # `viz_outputs` are now the only thing the caller needs.
            self._viz_buffer = None
            # Final VAE cache clear so the entry script doesn't see
            # stale internal state.
            self.vae.model.clear_cache()

        torch.cuda.synchronize(device)
        wall_clock: float = time.perf_counter() - t_start

        stats: dict[str, Any] = {
            "wall_clock_seconds": wall_clock,
            "text_encode_seconds": text_encode_seconds,
            "vae_encode_seconds": vae_encode_seconds,
            "rollout_seconds": rollout_seconds,
            "vae_decode_seconds": vae_decode_seconds,
            "viz_decode_seconds": viz_decode_seconds,
            "per_step_losses": per_step_losses,
            "viz": viz_outputs,  # Optional[dict[str, Tensor]] keyed by viz group label.
        }
        latents_out: Optional[torch.Tensor] = output.detach().cpu() if return_latents else None
        return video, latents_out, stats

    # ------------------------------------------------------------------
    # Viz helpers (used only when `tto.viz_trajectory=true`).
    # ------------------------------------------------------------------

    def _decode_viz_buffer(
        self,
        viz_buffer: list[dict[str, Any]],
        device: torch.device,
        model_dtype: torch.dtype,
        output_latents: Optional[torch.Tensor] = None,
        num_input_frames: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Group captured latents by (block, sub-timestep) and decode each
        group's epoch sequence into a single RGB clip.

        Returns a dict keyed by `block_{BB:02d}_subtimestep_{S}` whose
        values are `uint8` RGB tensors of shape `[T*rgb_per_latent, H, W, 3]`
        ready to be written by `torchvision.io.write_video`. The clip
        timeline is the epoch axis: each per-epoch latent (3 frames at the
        canonical config) contributes `3 * rgb_per_latent` RGB frames in
        the order they were captured (epoch 0, 1, ..., B-1, optional
        "final"). The companion `frames_per_epoch` metadata is included in
        a sidecar key so the entry script can chapter the timeline if
        desired.
        """
        # Group entries by (block, subtimestep).
        groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for entry in viz_buffer:
            key = (int(entry["block"]), int(entry["subtimestep"]))
            groups.setdefault(key, []).append(entry)

        # Stable epoch ordering: integers first (ascending), then "final".
        def _epoch_sort_key(entry: dict[str, Any]) -> tuple[int, int]:
            e = entry["epoch"]
            return (0, int(e)) if isinstance(e, int) else (1, 0)

        outputs: dict[str, torch.Tensor] = {}
        # Decode each epoch capture SEPARATELY (VAE cache cleared in
        # between) rather than concatenating the group into one long
        # pseudo-video. The Wan VAE is temporally causal: decoding a
        # concatenation would condition each chapter's first frames on
        # the PREVIOUS epoch's latents (they are the same 3 frames at a
        # different optimisation state, not a temporal continuation),
        # producing spurious motion blends at chapter boundaries and
        # non-uniform chapter lengths (1 + 4*(T-1) frames per decode).
        # Per-capture decode avoids the contamination; each capture is
        # additionally prefixed with the block's true preceding rollout
        # latents (see `viz_decode_context_frames` below) so chapters are
        # exactly `4 * num_frame_per_block` frames (12 at the canonical
        # config) decoded with the same causal context as the merged
        # video. Without an available prefix (T2V block 0, or the knob
        # set to 0) a chapter is `1 + 4*(num_frame_per_block - 1)` frames
        # (9) and its first frames decode from zero context.
        for (block_idx, subtimestep_idx), entries in sorted(groups.items()):
            entries_sorted = sorted(entries, key=_epoch_sort_key)

            # Decode context: the block's actual preceding latent frames
            # from the committed rollout. Blocks are generated
            # sequentially and never revisited, so `output_latents` at
            # end-of-generate equals its state at capture time — the
            # prefix is the SAME context the merged-video decode saw.
            # Without it, the causal VAE reconstructs the chapter's first
            # frames from zero context and they come out visibly worse.
            # All chapters of a block share one prefix, so differences
            # between chapters reflect only the TTO state.
            prefix: Optional[torch.Tensor] = None
            if output_latents is not None and self._viz_decode_context_frames > 0:
                block_abs_start = num_input_frames + block_idx * self.num_frame_per_block
                prefix_len = min(self._viz_decode_context_frames, block_abs_start)
                if prefix_len > 0:
                    prefix = output_latents[
                        :1, block_abs_start - prefix_len:block_abs_start
                    ].to(device=device, dtype=model_dtype)

            # ---- Pass 1: decode every chapter, keep CLEAN float frames
            #      on the GPU. Metrics and diff overlays must be computed
            #      BEFORE labels are burned in — the differing epoch
            #      labels would otherwise register as pixel change. ----
            chapters: list[dict[str, Any]] = []
            for e in entries_sorted:
                latent = e["latent"].to(device=device, dtype=model_dtype)
                num_capture_latents = int(latent.shape[1])
                if prefix is not None:
                    latent = torch.cat([prefix, latent], dim=1)
                with torch.no_grad():
                    rgb = self.vae(latent, mode="decode", use_cache=False)
                self.vae.model.clear_cache()
                rgb = (rgb * 0.5 + 0.5).clamp(0, 1)[0].float()  # [T, 3, H, W]
                if prefix is not None:
                    # A decode of T latents yields `1 + 4*(T-1)` RGB
                    # frames; the trailing `4 * num_capture_latents`
                    # belong to the capture — drop the prefix's frames.
                    rgb = rgb[-4 * num_capture_latents:]
                chapters.append({"epoch": e["epoch"], "frames": rgb})

            # ---- Pass 2: change metrics (+ optional diff overlays), all
            #      against the epoch-0 (unoptimised) reference chapter. ----
            ref_idx: int = next(
                (i for i, c in enumerate(chapters) if c["epoch"] == 0), 0
            )
            ref: torch.Tensor = chapters[ref_idx]["frames"]
            ref_gray_dim: Optional[torch.Tensor] = None
            if self._viz_diff_overlay:
                ref_gray = (
                    0.299 * ref[:, 0] + 0.587 * ref[:, 1] + 0.114 * ref[:, 2]
                )  # [T, H, W]
                ref_gray_dim = (ref_gray * 0.35).unsqueeze(-1).expand(-1, -1, -1, 3)

            metrics: list[dict[str, Any]] = []
            diff_chapters: list[torch.Tensor] = []
            for c in chapters:
                m = self._viz_change_metrics(c["frames"], ref)
                m["epoch"] = str(c["epoch"])
                metrics.append(m)
                if self._viz_diff_overlay:
                    diff_chapters.append(
                        self._viz_diff_chapter(c["frames"], ref, ref_gray_dim)
                    )

            # ---- Pass 3: quantise to uint8, burn labels, assemble. ----
            main_clips: list[torch.Tensor] = []
            for i, c in enumerate(chapters):
                clip_u8 = (
                    c["frames"].permute(0, 2, 3, 1).contiguous() * 255.0
                ).round().to("cpu", torch.uint8)
                ep = c["epoch"]
                ep_label = f"epoch {ep}" if isinstance(ep, int) else str(ep)
                if i == ref_idx:
                    ep_label += " (ref)"
                if self._viz_annotate:
                    self._viz_annotate_frames(
                        clip_u8,
                        f"block {block_idx:02d} | subtimestep {subtimestep_idx} "
                        f"| {ep_label} | dMAE {metrics[i]['mae255']:.2f}",
                    )
                    if self._viz_diff_overlay:
                        self._viz_annotate_frames(
                            diff_chapters[i],
                            f"block {block_idx:02d} | subtimestep {subtimestep_idx} "
                            f"| {ep_label} | diff scale {self._viz_diff_scale:.2f}",
                        )
                main_clips.append(clip_u8)
            del chapters

            key = f"block_{block_idx:02d}_subtimestep_{subtimestep_idx}"
            outputs[key] = torch.cat(main_clips, dim=0)
            if self._viz_diff_overlay:
                outputs[f"{key}__diff"] = torch.cat(diff_chapters, dim=0)  # type: ignore[assignment]
            # Sidecar metadata so the entry script can chapter the timeline
            # and rank clips without recomputing anything. `frames_per_epoch`
            # is exact and constant per group (equal latent lengths).
            epochs_in_order = [str(e["epoch"]) for e in entries_sorted]
            outputs[f"{key}__epochs"] = epochs_in_order  # type: ignore[assignment]
            outputs[f"{key}__frames_per_epoch"] = int(main_clips[0].shape[0])  # type: ignore[assignment]
            outputs[f"{key}__metrics"] = metrics  # type: ignore[assignment]
            # LAST chapter in epoch order ("final" when captured, else the
            # last TTO epoch) vs the reference = total TTO effect; the
            # entry script embeds it into the clip filename for triage.
            outputs[f"{key}__final_metrics"] = metrics[-1]  # type: ignore[assignment]
        return outputs

    def _viz_annotate_frames(self, frames_u8: torch.Tensor, label: str) -> None:
        """Stamp `label` into the top-left corner of every frame of
        `frames_u8` (`[T, H, W, 3]` uint8 on CPU), in place.

        The label is rasterised ONCE with PIL, then applied to the whole
        frame stack in a single masked write: the boxed region is darkened
        to 35% brightness and the text pixels set to white, so the label
        stays readable over any content and survives H.264 compression."""
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        height = int(frames_u8.shape[1])
        width = int(frames_u8.shape[2])
        if self._viz_font is None:
            size = max(16, height // 20)
            try:
                # Pillow >= 10.1 supports sizing the bundled default font.
                self._viz_font = ImageFont.load_default(size=size)
            except TypeError:
                self._viz_font = ImageFont.load_default()
        font = self._viz_font

        measurer = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        bbox = measurer.textbbox((0, 0), label, font=font)
        pad = 8
        box_w = min(bbox[2] - bbox[0] + 2 * pad, width)
        box_h = min(bbox[3] - bbox[1] + 2 * pad, height)

        patch = Image.new("L", (box_w, box_h), 0)
        ImageDraw.Draw(patch).text(
            (pad - bbox[0], pad - bbox[1]), label, fill=255, font=font
        )
        # [box_h, box_w, 1] bool — broadcasts over the T and channel axes.
        # `np.array` (not `asarray`) copies out of PIL's non-writable
        # buffer, avoiding torch's non-writable-tensor warning.
        text_mask = (
            torch.from_numpy(np.array(patch, dtype=np.uint8)) > 32
        ).unsqueeze(-1)

        region = frames_u8[:, :box_h, :box_w, :]
        darkened = (region.to(torch.float32) * 0.35).to(torch.uint8)
        frames_u8[:, :box_h, :box_w, :] = darkened.masked_fill(text_mask, 255)

    def _viz_change_metrics(
        self, frames: torch.Tensor, ref: torch.Tensor
    ) -> dict[str, Any]:
        """Change metrics between one chapter and the epoch-0 reference.

        Both inputs are `[T, 3, H, W]` float32 in [0, 1] on the same
        device; frame i is compared against reference frame i (chapters
        are the same content at different LoRA states, so frames align).

        Returned python floats:
            * `mae255`  — mean |delta| in 0-255 units. Linear and honest
              for near-identical frames, where PSNR saturates.
            * `p99_255` — 99th percentile of |delta| (0-255 units).
              Catches small-area strong edits the mean washes out.
            * `psnr`    — peak SNR in dB, capped at 99 (identical frames
              would be infinite).
            * `ssim` / `dssim` — structural similarity (Wang et al.,
              11x11 gaussian window, sigma 1.5) and its drop `1 - ssim`.
        """
        with torch.no_grad():
            d = frames - ref
            abs_d = d.abs()
            mae255 = float(abs_d.mean()) * 255.0
            flat = abs_d.flatten()
            k = max(1, int(round(0.99 * flat.numel())))
            p99_255 = float(flat.kthvalue(k).values) * 255.0
            mse = float((d * d).mean())
            psnr = 99.0 if mse < 1e-12 else min(99.0, -10.0 * math.log10(mse))
            ssim = float(self._ssim_video(frames, ref))
        return {
            "mae255": mae255,
            "p99_255": p99_255,
            "psnr": psnr,
            "ssim": ssim,
            "dssim": 1.0 - ssim,
        }

    def _ssim_video(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Mean SSIM over all frames/channels of two `[T, 3, H, W]`
        float stacks in [0, 1]. Canonical Wang et al. setup: 11x11
        gaussian window (sigma 1.5), C1 = 0.01^2, C2 = 0.03^2 (L = 1),
        per-channel maps averaged. The window is cached per device."""
        if (
            self._viz_ssim_window is None
            or self._viz_ssim_window.device != a.device
            or self._viz_ssim_window.dtype != a.dtype
        ):
            coords = torch.arange(11, dtype=a.dtype, device=a.device) - 5.0
            g = torch.exp(-(coords ** 2) / (2.0 * 1.5 ** 2))
            g = (g / g.sum()).unsqueeze(0)
            self._viz_ssim_window = (g.t() @ g).expand(3, 1, 11, 11).contiguous()
        w = self._viz_ssim_window
        conv = torch.nn.functional.conv2d
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_a = conv(a, w, padding=5, groups=3)
        mu_b = conv(b, w, padding=5, groups=3)
        mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
        # Clamp variances at 0: float error can produce tiny negatives.
        sig_a = (conv(a * a, w, padding=5, groups=3) - mu_a2).clamp_min(0.0)
        sig_b = (conv(b * b, w, padding=5, groups=3) - mu_b2).clamp_min(0.0)
        sig_ab = conv(a * b, w, padding=5, groups=3) - mu_ab
        ssim_map = ((2.0 * mu_ab + c1) * (2.0 * sig_ab + c2)) / (
            (mu_a2 + mu_b2 + c1) * (sig_a + sig_b + c2)
        )
        return ssim_map.mean()

    def _viz_diff_chapter(
        self,
        frames: torch.Tensor,
        ref: torch.Tensor,
        ref_gray_dim: torch.Tensor,
    ) -> torch.Tensor:
        """Single-colour absolute-difference chapter for one epoch.

        `mag` = channel-mean of |frames - ref| (the same reduction the
        dMAE metric uses). A pure-red overlay is alpha-blended over the
        dimmed grayscale reference with per-pixel opacity
        `mag / viz_diff_scale` (clamped) — fully transparent where
        nothing changed, saturated red at (or beyond) the fixed
        full-scale difference. Returns `[T, H, W, 3]` uint8 on CPU."""
        with torch.no_grad():
            mag = (frames - ref).abs().mean(dim=1)  # [T, H, W]
            alpha = (mag / self._viz_diff_scale).clamp(0.0, 1.0).unsqueeze(-1)
            red = torch.tensor(
                [1.0, 0.0, 0.0], dtype=frames.dtype, device=frames.device
            )
            out = (1.0 - alpha) * ref_gray_dim + alpha * red
            return (out * 255.0).round().to("cpu", torch.uint8)

    # ------------------------------------------------------------------
    # Inner TTO loops — the only places where the optimisations live.
    # ------------------------------------------------------------------

    def _optimize_block_step(
        self,
        tto_trainer: trainer.Trainer,
        state: SimpleNamespace,
        timestep: torch.Tensor,
        conditional_dict: dict[str, torch.Tensor],
        index: int,
        block_index: int,
        batch_size: int,
        current_num_frames: int,
        noise: torch.Tensor,
        num_input_frames: int,
    ) -> dict[int, dict[str, float]]:
        device: torch.device = noise.device
        measure_only: bool = bool(tto_trainer.config.tto.get("measure_only", False))

        # Idea 3 — `tto.optimize_at_subtimesteps` short-circuit. See the
        # reference TTO pipeline for the full rationale.
        optimize_at = tto_trainer.config.tto.get("optimize_at_subtimesteps", None)
        if optimize_at is not None and index not in list(optimize_at):
            return {}

        # LoRA reset policy: see `causal_inference_tto.py:_optimize_block_step`
        # for the rationale. Default `reset_actor_per_subtimestep=true`
        # preserves the historical per-(block, sub-timestep) reset; set
        # `false` to reset only at sub-timestep 0 (true per-block).
        if not measure_only and tto_trainer.config.tto.get("reset_actor_per_block", False):
            reset_per_substep: bool = bool(
                tto_trainer.config.tto.get("reset_actor_per_subtimestep", True)
            )
            if reset_per_substep or index == 0:
                tto_trainer.reset_model()

        # Two orthogonal noise-reuse knobs — see the reference TTO pipeline
        # for the full rationale.
        state.single_critic_noise = None
        state.last_critic_noise = None
        if (
            tto_trainer.config.tto.get("critic_all_epochs_single_noise", False)
            and index < len(self.denoising_step_list) - 1
        ):
            state.single_critic_noise = torch.randn(
                [batch_size, current_num_frames, noise.shape[2], noise.shape[3], noise.shape[4]],
                device=device, dtype=noise.dtype,
            )

        # --- Critic pre-compute: run in-place on the live cache. ---
        # We snapshot just enough state to undo whatever the model writes
        # (see `_save_kv_cache_for_inplace`). This covers both the
        # cache-has-room path and the shift-then-write rolling path
        # (`wan/modules/causal_model.py:206-222`) without falling back to a
        # full ~5.85 GB clone. The forward itself uses the existing
        # `tto_trainer.critic_model` so the operations and saved tensors
        # match the reference pipeline bit-for-bit.
        self._mem(f"    optimize_block_step({block_index},{index}) entry")
        with torch.no_grad():
            saved_kv = self._save_kv_cache_for_inplace(
                state.kv_cache1,
                current_start_frame=state.current_start_frame,
                num_new_frames=current_num_frames,
            )
            saved_ca = self._save_crossattn_is_init(state.crossattn_cache)
            _, critic_denoised_pred = tto_trainer.critic_model(
                noisy_image_or_video=state.noisy_input.detach(),
                conditional_dict=conditional_dict,
                timestep=timestep.detach(),
                kv_cache=state.kv_cache1,
                crossattn_cache=state.crossattn_cache,
                current_start=state.current_start_frame * self.FRAME_SEQ_LENGTH,
            )
            self._restore_kv_cache_for_inplace(state.kv_cache1, saved_kv)
            self._restore_crossattn_is_init(state.crossattn_cache, saved_ca)
            self._mem(f"    after critic_denoised_pred (in-place)")

        if measure_only:
            with torch.no_grad():
                loss_dict = self._measure_loss(
                    tto_trainer=tto_trainer,
                    state=state,
                    timestep=timestep,
                    conditional_dict=conditional_dict,
                    index=index,
                    block_index=block_index,
                    current_num_frames=current_num_frames,
                    noise=noise,
                    num_input_frames=num_input_frames,
                    critic_denoised_pred=critic_denoised_pred,
                )
            return {0: loss_dict}

        per_epoch_losses: dict[int, dict[str, float]] = {}
        for epoch in range(tto_trainer.tto_epochs):
            per_epoch_losses[epoch] = self._forward_backward(
                tto_trainer=tto_trainer,
                state=state,
                timestep=timestep,
                conditional_dict=conditional_dict,
                index=index,
                block_index=block_index,
                batch_size=batch_size,
                current_num_frames=current_num_frames,
                noise=noise,
                num_input_frames=num_input_frames,
                critic_denoised_pred=critic_denoised_pred,
                epoch=epoch,
            )
        return per_epoch_losses

    def _measure_loss(
        self,
        tto_trainer: trainer.Trainer,
        state: SimpleNamespace,
        timestep: torch.Tensor,
        conditional_dict: dict[str, torch.Tensor],
        index: int,
        block_index: int,
        current_num_frames: int,
        noise: torch.Tensor,
        num_input_frames: int,
        critic_denoised_pred: torch.Tensor,
    ) -> dict[str, float]:
        device: torch.device = noise.device

        # `_compute_loss` does up to 2 forwards on the cache (trainable +
        # critic look-ahead, plus a context fwd at the last sub-timestep).
        # If ANY of those would trigger the model's internal shift-then-write
        # rolling, we must fall back to cloning the whole cache — the
        # window-save trick can only undo writes inside its captured slice.
        n_writes = 3 if (index == len(self.denoising_step_list) - 1) else 2
        if self._would_overflow_after_writes(state.kv_cache1, n_writes):
            return self._compute_loss_via_clone(
                tto_trainer=tto_trainer,
                state=state,
                timestep=timestep,
                conditional_dict=conditional_dict,
                index=index,
                block_index=block_index,
                current_num_frames=current_num_frames,
                noise=noise,
                num_input_frames=num_input_frames,
                critic_denoised_pred=critic_denoised_pred,
                epoch=0,
            )[1]
        saved_kv = self._save_kv_cache_window(
            state.kv_cache1, n_frames_written=n_writes * current_num_frames
        )
        saved_ca = self._save_crossattn_is_init(state.crossattn_cache)
        critic_state = SimpleNamespace(
            noisy_input=state.noisy_input.detach().clone(),
            current_start_frame=state.current_start_frame,
            crossattn_cache=state.crossattn_cache,
            kv_cache1=state.kv_cache1,
            shared_next_noise=getattr(state, "single_critic_noise", None),
        )
        critic_timestep: torch.Tensor = timestep.detach().clone()
        _, loss_dict, _ = self._compute_loss(
            tto_trainer=tto_trainer,
            critic_state=critic_state,
            critic_timestep=critic_timestep,
            conditional_dict=conditional_dict,
            index=index,
            block_index=block_index,
            current_num_frames=current_num_frames,
            noise=noise,
            num_input_frames=num_input_frames,
            critic_denoised_pred=critic_denoised_pred,
            epoch=0,
        )
        self._restore_kv_cache_window(state.kv_cache1, saved_kv)
        self._restore_crossattn_is_init(state.crossattn_cache, saved_ca)
        return loss_dict

    def _forward_backward(
        self,
        tto_trainer: trainer.Trainer,
        state: SimpleNamespace,
        timestep: torch.Tensor,
        conditional_dict: dict[str, torch.Tensor],
        index: int,
        block_index: int,
        batch_size: int,
        current_num_frames: int,
        noise: torch.Tensor,
        num_input_frames: int,
        critic_denoised_pred: torch.Tensor,
        epoch: int,
    ) -> dict[str, float]:
        # The inner forward+backward path generates an autograd graph that
        # gradient checkpointing later re-runs. If we mutate the cache
        # in-place during the forward pass and then save/restore *after*
        # backward, the checkpoint recomputation sees a cache state that
        # differs from the original forward → it raises a shape/identity
        # mismatch. So this path *must* operate on a cloned cache.
        # Note: this revision still saves memory vs the reference impl via
        # (a) `lora.disabled(...)` instead of a separate critic model and
        # (b) skipping the `batch_repeat` allocation at B=1, both of which
        # are applied inside `_compute_loss`.
        self._mem(f"      forward_backward({block_index},{index},ep={epoch}) entry (cloned)")
        return self._forward_backward_via_clone(
            tto_trainer=tto_trainer,
            state=state,
            timestep=timestep,
            conditional_dict=conditional_dict,
            index=index,
            block_index=block_index,
            current_num_frames=current_num_frames,
            noise=noise,
            num_input_frames=num_input_frames,
            critic_denoised_pred=critic_denoised_pred,
            epoch=epoch,
        )

    # ------------------------------------------------------------------
    # Reference fall-back paths (kept identical to the cloned baseline).
    # ------------------------------------------------------------------

    def _forward_backward_via_clone(
        self,
        tto_trainer: trainer.Trainer,
        state: SimpleNamespace,
        timestep: torch.Tensor,
        conditional_dict: dict[str, torch.Tensor],
        index: int,
        block_index: int,
        current_num_frames: int,
        noise: torch.Tensor,
        num_input_frames: int,
        critic_denoised_pred: torch.Tensor,
        epoch: int,
    ) -> dict[str, float]:
        device: torch.device = noise.device
        critic_state = SimpleNamespace(
            noisy_input=state.noisy_input.detach().clone(),
            current_start_frame=state.current_start_frame,
            crossattn_cache=self._clone_crossattn_cache(state.crossattn_cache, device),
            kv_cache1=self._clone_kv_cache(state.kv_cache1, device),
            shared_next_noise=getattr(state, "single_critic_noise", None),
        )
        self._mem(f"      after per-iteration cache clones")
        critic_timestep: torch.Tensor = timestep.detach().clone()
        loss, loss_dict, used_noise = self._compute_loss(
            tto_trainer=tto_trainer,
            critic_state=critic_state,
            critic_timestep=critic_timestep,
            conditional_dict=conditional_dict,
            index=index,
            block_index=block_index,
            current_num_frames=current_num_frames,
            noise=noise,
            num_input_frames=num_input_frames,
            critic_denoised_pred=critic_denoised_pred,
            epoch=epoch,
        )
        # Record the noise the critic look-ahead consumed THIS iteration so
        # the outer `generate()` can reuse it via `tto.reuse_critic_noise`.
        if used_noise is not None:
            state.last_critic_noise = used_noise.detach()
        self._mem(f"      after _compute_loss")
        # `tto.update_every` (gradient accumulation): accumulate the
        # window-averaged gradient and step the optimizer only at window
        # boundaries. The final epoch also steps so a trailing partial
        # window is not discarded — keep `tto.epochs` a multiple of
        # `tto.update_every` for exact averaging. Default 1 preserves the
        # historical behaviour (backward + step every iteration).
        update_every: int = max(1, int(tto_trainer.config.tto.get("update_every", 1)))
        dbg_upd: bool = bool(tto_trainer.config.tto.get("debug_update_every", False))
        if update_every == 1:
            tto_trainer.backward_loss(loss)
        else:
            (loss / update_every).backward()
            if dbg_upd:
                gsq = sum(
                    float(p.grad.float().pow(2).sum())
                    for p in tto_trainer.model.parameters()
                    if p.requires_grad and p.grad is not None
                )
                wsq = sum(
                    float(p.data.float().pow(2).sum())
                    for p in tto_trainer.model.parameters() if p.requires_grad
                )
                print(
                    f"[upd-dbg] epoch {epoch}: accum_grad_norm={gsq ** 0.5:.3e}"
                    f" weight_norm={wsq ** 0.5:.6e}",
                    flush=True,
                )
            if (epoch + 1) % update_every == 0 or epoch == tto_trainer.tto_epochs - 1:
                tto_trainer.optimizer.step()
                if dbg_upd:
                    wsq2 = sum(
                        float(p.data.float().pow(2).sum())
                        for p in tto_trainer.model.parameters() if p.requires_grad
                    )
                    print(
                        f"[upd-dbg] epoch {epoch}: STEP fired;"
                        f" weight_norm after={wsq2 ** 0.5:.6e}",
                        flush=True,
                    )
        self._mem(f"      after backward + optimizer.step()")
        return loss_dict

    def _compute_loss_via_clone(
        self, **kwargs,
    ) -> "tuple[torch.Tensor, dict[str, float], Optional[torch.Tensor]]":
        device = kwargs["noise"].device
        state = kwargs.pop("state")
        critic_state = SimpleNamespace(
            noisy_input=state.noisy_input.detach().clone(),
            current_start_frame=state.current_start_frame,
            crossattn_cache=self._clone_crossattn_cache(state.crossattn_cache, device),
            kv_cache1=self._clone_kv_cache(state.kv_cache1, device),
            shared_next_noise=getattr(state, "single_critic_noise", None),
        )
        critic_timestep: torch.Tensor = kwargs.pop("timestep").detach().clone()
        return self._compute_loss(
            critic_state=critic_state,
            critic_timestep=critic_timestep,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core loss computation. Uses `lora.disabled(...)` instead of the
    # `tto_trainer.critic_model` deep-copy.
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        tto_trainer: trainer.Trainer,
        critic_state: SimpleNamespace,
        critic_timestep: torch.Tensor,
        conditional_dict: dict[str, torch.Tensor],
        index: int,
        block_index: int,
        current_num_frames: int,
        noise: torch.Tensor,
        num_input_frames: int,
        critic_denoised_pred: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # `tto.update_every` (gradient accumulation): zero the grads only
        # at the START of each accumulation window. Iterations inside a
        # window accumulate `loss / update_every` gradients and the
        # optimizer steps once at the window's end (see
        # `_forward_backward_via_clone`). Default 1 reproduces the
        # historical per-iteration update byte-identically.
        update_every: int = max(1, int(tto_trainer.config.tto.get("update_every", 1)))
        if epoch % update_every == 0:
            tto_trainer.optimizer.zero_grad(set_to_none=True)
            if tto_trainer.config.tto.get("debug_update_every", False):
                print(f"[upd-dbg] zero_grad at epoch {epoch}", flush=True)

        opt_batch_size: int = tto_trainer.config.tto.get("batch_size", 1)
        device: torch.device = noise.device
        self._mem(f"        _compute_loss entry (B={opt_batch_size})")

        # ---- Lookahead cache-gradient fix (revertible). ----
        # In the `next_block_*`/`dual` modes the ONLY route from the
        # gaussian loss back to the LoRA runs through in-place KV-cache
        # writes consumed by a LATER model call (the look-ahead's own
        # input is pure next-block noise, which carries no gradient).
        # Gradient checkpointing severs that route — verified
        # empirically: bit-identical losses across all epochs with
        # loss_reg pinned at exactly 0 (the LoRA never leaves its reset
        # state). While the fix is active we:
        #   * disable gradient checkpointing on the forwards whose cache
        #     writes/reads must stay differentiable — the critic (inject
        #     + look-ahead) always; the trainable too under
        #     `next_block_no_inject`/`dual` (their look-ahead reads the
        #     trainable's own writes; under `next_block_inject` the
        #     inject overwrites them, so the trainable may stay
        #     checkpointed);
        #   * enable `cache_grad_safe_reads` (clone-on-read) on those
        #     models so tensors saved for the attention backward survive
        #     later in-place cache mutations (version-counter hazard);
        #     cleared for the FINAL look-ahead, whose saves are never
        #     invalidated (backward follows immediately).
        # Gate: `tto.lookahead_cache_grad_fix` (default True). Setting
        # it to false restores the previous (gradient-dead) behaviour.
        lookahead_target: str = str(
            tto_trainer.config.tto.get("lookahead_target", "next_subtimestep")
        )
        fix_active: bool = (
            lookahead_target in ("next_block_inject", "next_block_no_inject", "dual")
            and bool(tto_trainer.config.tto.get("lookahead_cache_grad_fix", True))
            and torch.is_grad_enabled()
        )
        fix_wrappers: list = []
        if fix_active:
            # Un-checkpoint BOTH models in every fixed mode. Keeping the
            # trainable checkpointed (earlier inject-mode design) makes
            # every backward re-run its blocks, and each recompute
            # re-writes the (cloned) cache in place — version-bumping the
            # other forwards' saved tensors. Un-checkpointing eliminates
            # the recompute-mutation class entirely; clone-on-read covers
            # the remaining forward-time mutations (inject overwrite,
            # sink-EMA blend, rolling shifts).
            fix_wrappers = [tto_trainer.critic_model, tto_trainer.model]
            for w in fix_wrappers:
                w.disable_gradient_checkpointing()
                self._set_cache_grad_safe_reads(w, True)

        # Trainable forward.
        _, denoised_pred = tto_trainer.model(
            noisy_image_or_video=critic_state.noisy_input,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
            kv_cache=critic_state.kv_cache1,
            crossattn_cache=critic_state.crossattn_cache,
            current_start=critic_state.current_start_frame * self.FRAME_SEQ_LENGTH,
        )
        self._mem(f"        after trainable fwd")
        # Viz capture (no-op when `tto.viz_trajectory=false`). The captured
        # `denoised_pred` uses the LoRA state BEFORE this epoch's backward,
        # so "epoch 0" is the unoptimised model. Under gradient
        # accumulation (`tto.update_every>1`) the LoRA only changes at
        # window boundaries, so capture only the FIRST iteration of each
        # window — one chapter per OPTIMIZATION step (plus "final"); the
        # intermediate accumulation iterations would be identical frames.
        if epoch % update_every == 0:
            self._viz_record(denoised_pred, block_index, index, epoch)

        loss_reg: torch.Tensor = torch.nn.functional.mse_loss(
            denoised_pred, critic_denoised_pred
        )

        # Skip the no-op `repeat((1,1,1,1))` allocation at B=1. The flag is
        # exposed via `tto.skip_batch_repeat_at_b1` for bisection; default
        # is True (skip the allocation).
        skip_b1 = bool(tto_trainer.config.tto.get("skip_batch_repeat_at_b1", True))
        if opt_batch_size > 1 or not skip_b1:
            critic_state.crossattn_cache = self._crossattn_cache_batch_expand(
                critic_state.crossattn_cache, opt_batch_size
            )
            critic_state.kv_cache1 = self._kv_cache_batch_repeat(
                critic_state.kv_cache1, opt_batch_size
            )
            _, t, c, h, w = denoised_pred.size()
            denoised_pred_b: torch.Tensor = denoised_pred.expand((opt_batch_size, t, c, h, w))
            cond_dict_b: dict[str, torch.Tensor] = {
                "prompt_embeds": conditional_dict["prompt_embeds"].expand(opt_batch_size, -1, -1)
            }
        else:
            _, t, c, h, w = denoised_pred.size()
            denoised_pred_b = denoised_pred
            cond_dict_b = conditional_dict
        self._mem(f"        after batch-expand (B={opt_batch_size})")

        # `lookahead_target` picks the critic look-ahead's temporal target:
        #   * `next_subtimestep` (default): existing behaviour. Within-block
        #     look-ahead at `t[index+1]` for `index < last`, next-block
        #     look-ahead with context-noise cache injection at `index == last`.
        #   * `next_block_no_inject` (Variant 1b): always look one TEMPORAL
        #     block ahead at `t[0]`, NO context-noise cache injection. Same
        #     compute cost as the default; cache state at the current block
        #     range remains the trainable's K/V (not faithful to actual
        #     generation but cheap and clean signal).
        #   * `next_block_inject` (Variant 1a): always look one temporal
        #     block ahead at `t[0]`, WITH context-noise injection so the
        #     cache state matches generation. +1 critic fwd per sub-timestep
        #     (~3-4 GB extra peak via gradient_checkpointing).
        #   * `dual`: BOTH within-block and next-block-no-inject look-aheads
        #     at every sub-timestep where applicable. K/V at the current
        #     block range is snapshotted between the two look-aheads so the
        #     next-block one reads the trainable's K/V (not the within-
        #     block-look-ahead's overwrite). At the last sub-timestep
        #     within-block has no target — falls through to next-block-no-
        #     inject only. Combined as `wb_w * gauss_wb + nb_w * gauss_nb`.
        outer_batch: int = int(noise.shape[0])
        used_noise: Optional[torch.Tensor] = None
        # (`lookahead_target` and the cache-gradient fix state were
        # resolved at the top of this method.)
        is_last_subtimestep: bool = (index >= len(self.denoising_step_list) - 1)

        if lookahead_target == "dual":
            # ---- Dual look-ahead: within-block (if not last) + next-block-no-inject. ----
            wb_weight = float(tto_trainer.config.tto.get("lookahead_within_block_weight", 1.0))
            nb_weight = float(tto_trainer.config.tto.get("lookahead_next_block_weight", 1.0))
            per_moment_loss: dict[str, float] = {}
            gauss_wb: Optional[torch.Tensor] = None
            gauss_nb: torch.Tensor
            critic_next_timestep_for_log: int

            if not is_last_subtimestep:
                # Snapshot trainable's K/V at the current block range so the
                # within-block look-ahead's in-place overwrites can be undone
                # before the next-block look-ahead reads the cache.
                kv_snap = self._snapshot_kv_at_last_block(
                    critic_state.kv_cache1, current_num_frames
                )

                # Within-block look-ahead: re-noise to t[index+1].
                wb_next_t = float(self.denoising_step_list[index + 1])
                shared = getattr(critic_state, "shared_next_noise", None)
                if shared is not None:
                    next_noise_wb = (
                        shared.expand_as(denoised_pred_b)
                        if shared.shape[0] != denoised_pred_b.shape[0] else shared
                    )
                else:
                    next_noise_wb = torch.randn_like(denoised_pred_b)
                wb_input = self.scheduler.add_noise(
                    denoised_pred_b.flatten(0, 1),
                    next_noise_wb.flatten(0, 1),
                    torch.full(
                        [opt_batch_size * current_num_frames],
                        wb_next_t, device=device, dtype=torch.float32,
                    ),
                ).unflatten(0, denoised_pred_b.shape[:2])
                wb_t_tensor = torch.full(
                    [opt_batch_size, current_num_frames], wb_next_t,
                    device=device, dtype=torch.float32,
                )
                _, _, pred_noise_wb = tto_trainer.critic_model(
                    noisy_image_or_video=wb_input,
                    conditional_dict=cond_dict_b,
                    timestep=wb_t_tensor,
                    kv_cache=critic_state.kv_cache1,
                    crossattn_cache=critic_state.crossattn_cache,
                    current_start=critic_state.current_start_frame * self.FRAME_SEQ_LENGTH,
                    return_pred_noise=True,
                )
                self._mem(f"        after dual wb-lookahead fwd")
                gauss_wb, per_moment_wb = tto_trainer.loss_fn(pred_noise_wb)
                per_moment_loss.update({f"{k}_wb": v for k, v in per_moment_wb.items()})

                # Restore K/V so the next-block look-ahead sees the trainable's
                # K/V at the current block range, not the within-block-look-
                # ahead's.
                self._restore_kv_at_last_block(critic_state.kv_cache1, kv_snap)

            # Next-block-no-inject look-ahead.
            if critic_state.current_start_frame < noise.size(dim=1):
                critic_state.current_start_frame += current_num_frames
            nb_next_t = float(self.denoising_step_list[0])
            nb_input = noise[
                :,
                critic_state.current_start_frame - num_input_frames - current_num_frames:
                critic_state.current_start_frame - num_input_frames,
            ].expand((opt_batch_size, t, c, h, w))
            self._maybe_roll_kv_cache_for_overflow(critic_state)
            nb_t_tensor = torch.full(
                [opt_batch_size, current_num_frames], nb_next_t,
                device=device, dtype=torch.float32,
            )
            # NOTE: clone-on-read stays ON for this forward too — its
            # per-layer cache writes happen before its own reads, and any
            # later mutation (dual's snapshot restore) would invalidate
            # view saves.
            _, _, pred_noise_nb = tto_trainer.critic_model(
                noisy_image_or_video=nb_input,
                conditional_dict=cond_dict_b,
                timestep=nb_t_tensor,
                kv_cache=critic_state.kv_cache1,
                crossattn_cache=critic_state.crossattn_cache,
                current_start=critic_state.current_start_frame * self.FRAME_SEQ_LENGTH,
                return_pred_noise=True,
            )
            self._mem(f"        after dual nb-lookahead fwd")
            gauss_nb, per_moment_nb = tto_trainer.loss_fn(pred_noise_nb)
            per_moment_loss.update({f"{k}_nb": v for k, v in per_moment_nb.items()})

            if gauss_wb is not None:
                gaussian_loss = wb_weight * gauss_wb + nb_weight * gauss_nb
                critic_next_timestep_for_log = nb_next_t  # report the temporally-meaningful one
            else:
                # At last sub-timestep, dual degenerates to next-block-no-inject.
                gaussian_loss = gauss_nb
                critic_next_timestep_for_log = nb_next_t

            reg_w: float = float(tto_trainer.config.tto.regularization_weight)
            gauss_w: float = float(tto_trainer.config.tto.white_gaussian_noise_weight)
            total_loss: torch.Tensor = gauss_w * gaussian_loss + reg_w * loss_reg
            loss_dict: dict[str, float] = {
                "loss_total": total_loss.detach().cpu().item(),
                "loss_gaussian": gaussian_loss.detach().cpu().item(),
                "loss_reg": loss_reg.detach().cpu().item(),
                "regularization_weight": reg_w,
                "gaussian_weight": gauss_w,
                "critic_timestep": float(critic_next_timestep_for_log),
                "lookahead_target": lookahead_target,
                **per_moment_loss,
            }
            if gauss_wb is not None:
                loss_dict["loss_gaussian_wb"] = gauss_wb.detach().cpu().item()
                loss_dict["loss_gaussian_nb"] = gauss_nb.detach().cpu().item()
                loss_dict["lookahead_within_block_weight"] = wb_weight
                loss_dict["lookahead_next_block_weight"] = nb_weight
            # In dual mode the within-block sampled noise (if any) is the
            # only candidate to thread back via `reuse_critic_noise`, but
            # since the temporal next-block branch also contributes to the
            # loss, the semantics of "reuse" become ambiguous. Set None and
            # treat `reuse_critic_noise` as inert under `dual`.
            self._lookahead_fix_restore(tto_trainer, fix_wrappers)
            return total_loss, loss_dict, None

        # ---- Single look-ahead modes (next_subtimestep / next_block_*). ----
        if lookahead_target == "next_subtimestep" and not is_last_subtimestep:
            # Within-block look-ahead at t[index+1].
            critic_next_timestep: float = float(self.denoising_step_list[index + 1])
            shared = getattr(critic_state, "shared_next_noise", None)
            if shared is not None:
                if shared.shape[0] != denoised_pred_b.shape[0]:
                    next_noise = shared.expand_as(denoised_pred_b)
                else:
                    next_noise = shared
                used_noise = shared
            else:
                next_noise = torch.randn_like(denoised_pred_b)
                used_noise = (
                    next_noise[:outer_batch].detach().clone()
                    if opt_batch_size > outer_batch else next_noise
                )
            critic_state.noisy_input = self.scheduler.add_noise(
                denoised_pred_b.flatten(0, 1),
                next_noise.flatten(0, 1),
                torch.full(
                    [opt_batch_size * current_num_frames],
                    critic_next_timestep,
                    device=device,
                    dtype=torch.float32,
                ),
            ).unflatten(0, denoised_pred_b.shape[:2])
        else:
            # Next-block look-ahead. Three sub-cases:
            #   * `next_subtimestep` at last sub-timestep — inject context-noise
            #     into the cache (existing behaviour).
            #   * `next_block_inject` (any sub-timestep) — inject context-noise.
            #   * `next_block_no_inject` (any sub-timestep) — skip the injection.
            inject_context_noise: bool = (
                lookahead_target == "next_block_inject"
                or (lookahead_target == "next_subtimestep" and is_last_subtimestep)
            )
            if inject_context_noise:
                context_timestep_b: torch.Tensor = torch.full(
                    [opt_batch_size, current_num_frames],
                    self.args.context_noise,
                    device=device,
                    dtype=torch.int64,
                )
                tto_trainer.critic_model(
                    noisy_image_or_video=denoised_pred_b,
                    conditional_dict=cond_dict_b,
                    timestep=context_timestep_b,
                    kv_cache=critic_state.kv_cache1,
                    crossattn_cache=critic_state.crossattn_cache,
                    current_start=critic_state.current_start_frame * self.FRAME_SEQ_LENGTH,
                )
            if critic_state.current_start_frame < noise.size(dim=1):
                critic_state.current_start_frame += current_num_frames
            critic_next_timestep = float(self.denoising_step_list[0])
            critic_state.noisy_input = noise[
                :,
                critic_state.current_start_frame - num_input_frames - current_num_frames:
                critic_state.current_start_frame - num_input_frames,
            ].expand((opt_batch_size, t, c, h, w))
            self._maybe_roll_kv_cache_for_overflow(critic_state)

        critic_timestep_next: torch.Tensor = torch.full(
            [opt_batch_size, current_num_frames],
            critic_next_timestep,
            device=device,
            dtype=torch.float32,
        )
        # NOTE: clone-on-read stays ON for the final look-ahead too — its
        # own per-layer writes precede its reads, and keeping clones makes
        # its saves immune to any later in-place mutation.
        _, _, pred_noise = tto_trainer.critic_model(
            noisy_image_or_video=critic_state.noisy_input,
            conditional_dict=cond_dict_b,
            timestep=critic_timestep_next,
            kv_cache=critic_state.kv_cache1,
            crossattn_cache=critic_state.crossattn_cache,
            current_start=critic_state.current_start_frame * self.FRAME_SEQ_LENGTH,
            return_pred_noise=True,
        )
        self._mem(f"        after critic look-ahead fwd")

        gaussian_loss, per_moment_loss = tto_trainer.loss_fn(pred_noise)
        reg_w: float = float(tto_trainer.config.tto.regularization_weight)
        gauss_w: float = float(tto_trainer.config.tto.white_gaussian_noise_weight)
        total_loss: torch.Tensor = gauss_w * gaussian_loss + reg_w * loss_reg

        loss_dict: dict[str, float] = {
            "loss_total": total_loss.detach().cpu().item(),
            "loss_gaussian": gaussian_loss.detach().cpu().item(),
            "loss_reg": loss_reg.detach().cpu().item(),
            "regularization_weight": reg_w,
            "gaussian_weight": gauss_w,
            "critic_timestep": float(critic_timestep_next[0, 0].detach().cpu().item()),
            "lookahead_target": lookahead_target,
            **per_moment_loss,
        }
        if bool(tto_trainer.config.tto.get("debug_lookahead_fix", False)):
            # Temporary diagnostic (config-gated): does the graph actually
            # reach the LoRA params from the gaussian term? Print header
            # FIRST and probe only the gaussian loss — probing multiple
            # terms re-walks the graph and, with any checkpointed
            # segment, each re-walk's recompute mutates the cache and
            # version-poisons the next probe.
            print(
                f"[lookahead-debug] blk={block_index} sub={index} ep={epoch} "
                f"fix_active={fix_active} "
                f"gen_ckpt={getattr(tto_trainer.model.model, 'gradient_checkpointing', '?')} "
                f"critic_ckpt={getattr(tto_trainer.critic_model.model, 'gradient_checkpointing', '?')} "
                f"gauss_requires_grad={gaussian_loss.requires_grad}",
                flush=True,
            )
            lora_params = [
                p for _, p in tto_trainer.model.named_parameters() if p.requires_grad
            ]
            gnorm = None
            dpred_norm = None
            n_unused = -1
            if gaussian_loss.requires_grad:
                # Split the chain: gradient at denoised_pred tells whether
                # the zero enters upstream (lookahead/inject) or
                # downstream (trainable) of the trainable's output.
                gd = torch.autograd.grad(
                    gaussian_loss, denoised_pred, retain_graph=True,
                    allow_unused=True,
                )[0]
                dpred_norm = None if gd is None else float(gd.abs().sum())
                g = torch.autograd.grad(
                    gaussian_loss, lora_params, retain_graph=True,
                    allow_unused=True,
                )
                n_unused = sum(1 for x in g if x is None)
                vals = [float(x.abs().sum()) for x in g if x is not None]
                gnorm = sum(vals) if vals else None
            print(
                f"[lookahead-debug]   gauss_grad_sum={gnorm} "
                f"unused={n_unused}/{len(lora_params)} "
                f"grad_at_denoised_pred={dpred_norm}",
                flush=True,
            )
        self._lookahead_fix_restore(tto_trainer, fix_wrappers)
        return total_loss, loss_dict, used_noise

    # ------------------------------------------------------------------
    # Idea 4 — per-block (trajectory-end) TTO. Memory-optimised variant.
    # Critic precompute uses `_save_kv_cache_for_inplace` /
    # `_restore_kv_cache_for_inplace` instead of cloning (~5.85 GB saved).
    # Per-epoch trainable trajectory still clones the cache because
    # gradient checkpointing's backward replay needs the pre-trajectory
    # state to be available untouched.
    # ------------------------------------------------------------------

    def _optimize_block_trajectory(
        self,
        tto_trainer: trainer.Trainer,
        state: SimpleNamespace,
        conditional_dict: dict[str, torch.Tensor],
        block_index: int,
        batch_size: int,
        current_num_frames: int,
        noise: torch.Tensor,
        num_input_frames: int,
    ) -> dict[int, dict[str, float]]:
        """Memory-optimised idea 4. See `causal_inference_tto.py`'s
        `_optimize_block_trajectory` for the math. The only structural
        difference is the frozen-critic precompute: instead of cloning
        the live cache for the 4 critic forwards, we snapshot just the
        regions the model will mutate (`_save_kv_cache_for_inplace`),
        let the 4 critic forwards run in-place on the live cache, then
        restore. All 4 critic forwards happen at the SAME
        current_start_frame, so only forward 1 may trigger rolling;
        forwards 2-4 take the no-roll branch and overwrite at the same
        write region, which means the single save/restore captures the
        whole transition correctly."""
        device: torch.device = noise.device
        measure_only: bool = bool(tto_trainer.config.tto.get("measure_only", False))
        num_subtimesteps: int = len(self.denoising_step_list)
        num_transitions: int = num_subtimesteps - 1

        if not measure_only and tto_trainer.config.tto.get("reset_actor_per_block", False):
            tto_trainer.reset_model()

        use_single_noise: bool = bool(
            tto_trainer.config.tto.get("critic_all_epochs_single_noise", False)
        )
        state.single_trajectory_noises = None
        state.last_trajectory_noises = None
        if use_single_noise and num_transitions > 0:
            state.single_trajectory_noises = [
                torch.randn(
                    [batch_size, current_num_frames,
                     noise.shape[2], noise.shape[3], noise.shape[4]],
                    device=device, dtype=noise.dtype,
                )
                for _ in range(num_transitions)
            ]

        # --- Frozen-critic precompute (in-place on the live cache). ---
        self._mem(f"    optimize_block_trajectory({block_index}) entry")
        with torch.no_grad():
            saved_kv = self._save_kv_cache_for_inplace(
                state.kv_cache1,
                current_start_frame=state.current_start_frame,
                num_new_frames=current_num_frames,
            )
            saved_ca = self._save_crossattn_is_init(state.crossattn_cache)
            self._mem(f"    after critic precompute snapshot (in-place)")

            critic_traj_noises: list[torch.Tensor] = (
                state.single_trajectory_noises
                if state.single_trajectory_noises is not None
                else [
                    torch.randn(
                        [batch_size, current_num_frames,
                         noise.shape[2], noise.shape[3], noise.shape[4]],
                        device=device, dtype=noise.dtype,
                    )
                    for _ in range(num_transitions)
                ]
            )
            critic_noisy: torch.Tensor = state.noisy_input.detach()
            critic_x0_final: Optional[torch.Tensor] = None
            for idx, t_val in enumerate(self.denoising_step_list):
                t_tensor = torch.full(
                    [batch_size, current_num_frames], float(t_val),
                    device=device, dtype=torch.float32,
                )
                _, critic_x0 = tto_trainer.critic_model(
                    noisy_image_or_video=critic_noisy,
                    conditional_dict=conditional_dict,
                    timestep=t_tensor,
                    kv_cache=state.kv_cache1,
                    crossattn_cache=state.crossattn_cache,
                    current_start=state.current_start_frame * self.FRAME_SEQ_LENGTH,
                )
                if idx < num_transitions:
                    t_next = float(self.denoising_step_list[idx + 1])
                    critic_noisy = self.scheduler.add_noise(
                        critic_x0.flatten(0, 1),
                        critic_traj_noises[idx].flatten(0, 1),
                        torch.full(
                            [batch_size * current_num_frames], t_next,
                            device=device, dtype=torch.float32,
                        ),
                    ).unflatten(0, critic_x0.shape[:2])
                else:
                    critic_x0_final = critic_x0
            # Restore the live cache to its pre-precompute state. The 4
            # critic forwards all wrote to the SAME position (block i),
            # so the single restore correctly reverts to the start state.
            self._restore_kv_cache_for_inplace(state.kv_cache1, saved_kv)
            self._restore_crossattn_is_init(state.crossattn_cache, saved_ca)
            self._mem(f"    after critic trajectory + restore")
        assert critic_x0_final is not None

        if measure_only:
            with torch.no_grad():
                _, loss_dict = self._compute_trajectory_loss(
                    tto_trainer=tto_trainer,
                    state=state,
                    conditional_dict=conditional_dict,
                    block_index=block_index,
                    batch_size=batch_size,
                    current_num_frames=current_num_frames,
                    noise=noise,
                    num_input_frames=num_input_frames,
                    critic_x0_final=critic_x0_final,
                    use_single_noise=use_single_noise,
                    epoch=0,
                    record_last_noises=False,
                )
            return {0: loss_dict}

        per_epoch_losses: dict[int, dict[str, float]] = {}
        for epoch in range(tto_trainer.tto_epochs):
            self._mem(f"    block_trajectory({block_index},ep={epoch}) entry")
            loss, loss_dict = self._compute_trajectory_loss(
                tto_trainer=tto_trainer,
                state=state,
                conditional_dict=conditional_dict,
                block_index=block_index,
                batch_size=batch_size,
                current_num_frames=current_num_frames,
                noise=noise,
                num_input_frames=num_input_frames,
                critic_x0_final=critic_x0_final,
                use_single_noise=use_single_noise,
                epoch=epoch,
                record_last_noises=(epoch == tto_trainer.tto_epochs - 1),
            )
            tto_trainer.backward_loss(loss)
            self._mem(f"    after backward + optimizer.step()")
            per_epoch_losses[epoch] = loss_dict
        return per_epoch_losses

    def _compute_trajectory_loss(
        self,
        tto_trainer: trainer.Trainer,
        state: SimpleNamespace,
        conditional_dict: dict[str, torch.Tensor],
        block_index: int,
        batch_size: int,
        current_num_frames: int,
        noise: torch.Tensor,
        num_input_frames: int,
        critic_x0_final: torch.Tensor,
        use_single_noise: bool,
        epoch: int,
        record_last_noises: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Same math as the reference variant's `_compute_trajectory_loss`.
        The cache is cloned here too (autograd requires it). The
        memory savings vs the reference variant come from the in-place
        critic precompute in `_optimize_block_trajectory` only.

        NOTE: an earlier revision wrapped this in
        `torch.utils.checkpoint.checkpoint(use_reentrant=False)` to
        recompute the 4-fwd activations during backward. That leaked
        ~8 GB per epoch (each epoch's checkpoint saved-input snapshot
        appears to retain a reference to the KV cache list-of-dicts that
        the GC couldn't clear). Reverted to the straightforward
        clone-based path; peak stays at ~32 GB."""
        device: torch.device = noise.device
        num_subtimesteps: int = len(self.denoising_step_list)
        num_transitions: int = num_subtimesteps - 1

        tto_trainer.optimizer.zero_grad(set_to_none=True)

        critic_state = SimpleNamespace(
            noisy_input=state.noisy_input.detach().clone(),
            current_start_frame=state.current_start_frame,
            crossattn_cache=self._clone_crossattn_cache(state.crossattn_cache, device),
            kv_cache1=self._clone_kv_cache(state.kv_cache1, device),
        )
        self._mem(f"      after per-iteration cache clones")

        if use_single_noise:
            transition_noises: list[torch.Tensor] = state.single_trajectory_noises  # type: ignore
        else:
            transition_noises = [
                torch.randn(
                    [batch_size, current_num_frames,
                     noise.shape[2], noise.shape[3], noise.shape[4]],
                    device=device, dtype=noise.dtype,
                )
                for _ in range(num_transitions)
            ]

        noisy_input: torch.Tensor = critic_state.noisy_input
        trainable_x0_final: Optional[torch.Tensor] = None
        for idx, t_val in enumerate(self.denoising_step_list):
            t_tensor = torch.full(
                [batch_size, current_num_frames], float(t_val),
                device=device, dtype=torch.float32,
            )
            _, denoised_pred = tto_trainer.model(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=t_tensor,
                kv_cache=critic_state.kv_cache1,
                crossattn_cache=critic_state.crossattn_cache,
                current_start=critic_state.current_start_frame * self.FRAME_SEQ_LENGTH,
            )
            if idx < num_transitions:
                t_next = float(self.denoising_step_list[idx + 1])
                noisy_input = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    transition_noises[idx].flatten(0, 1),
                    torch.full(
                        [batch_size * current_num_frames], t_next,
                        device=device, dtype=torch.float32,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                trainable_x0_final = denoised_pred
        assert trainable_x0_final is not None
        self._mem(f"      after trainable trajectory")

        loss_reg: torch.Tensor = torch.nn.functional.mse_loss(
            trainable_x0_final, critic_x0_final
        )

        context_timestep: torch.Tensor = torch.full(
            [batch_size, current_num_frames], self.args.context_noise,
            device=device, dtype=torch.int64,
        )
        tto_trainer.critic_model(
            noisy_image_or_video=trainable_x0_final,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=critic_state.kv_cache1,
            crossattn_cache=critic_state.crossattn_cache,
            current_start=critic_state.current_start_frame * self.FRAME_SEQ_LENGTH,
        )
        if critic_state.current_start_frame < noise.size(dim=1):
            critic_state.current_start_frame += current_num_frames

        critic_next_timestep: float = float(self.denoising_step_list[0])
        critic_state.noisy_input = noise[
            :,
            critic_state.current_start_frame - num_input_frames - current_num_frames:
            critic_state.current_start_frame - num_input_frames,
        ]
        self._maybe_roll_kv_cache_for_overflow(critic_state)

        critic_timestep_next: torch.Tensor = torch.full(
            [batch_size, current_num_frames], critic_next_timestep,
            device=device, dtype=torch.float32,
        )
        _, _, pred_noise = tto_trainer.critic_model(
            noisy_image_or_video=critic_state.noisy_input,
            conditional_dict=conditional_dict,
            timestep=critic_timestep_next,
            kv_cache=critic_state.kv_cache1,
            crossattn_cache=critic_state.crossattn_cache,
            current_start=critic_state.current_start_frame * self.FRAME_SEQ_LENGTH,
            return_pred_noise=True,
        )
        self._mem(f"      after end-of-trajectory critic look-ahead")

        gaussian_loss, per_moment_loss = tto_trainer.loss_fn(pred_noise)
        reg_w: float = float(tto_trainer.config.tto.regularization_weight)
        gauss_w: float = float(tto_trainer.config.tto.white_gaussian_noise_weight)
        total_loss: torch.Tensor = gauss_w * gaussian_loss + reg_w * loss_reg

        loss_dict: dict[str, float] = {
            "loss_total": total_loss.detach().cpu().item(),
            "loss_gaussian": gaussian_loss.detach().cpu().item(),
            "loss_reg": loss_reg.detach().cpu().item(),
            "regularization_weight": reg_w,
            "gaussian_weight": gauss_w,
            "critic_timestep": float(critic_timestep_next[0, 0].detach().cpu().item()),
            **per_moment_loss,
        }

        if record_last_noises:
            state.last_trajectory_noises = [n.detach() for n in transition_noises]

        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # KV-cache initialization & rolling helpers (unchanged from reference).
    # ------------------------------------------------------------------

    def _initialize_kv_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        kv_cache_size: int = (
            self.local_attn_size * self.FRAME_SEQ_LENGTH if self.local_attn_size != -1 else 32760
        )
        self.kv_cache1 = [
            {
                "k": torch.zeros(
                    [batch_size, kv_cache_size, 12, 128],
                    dtype=dtype, device=device, requires_grad=False,
                ),
                "v": torch.zeros(
                    [batch_size, kv_cache_size, 12, 128],
                    dtype=dtype, device=device, requires_grad=False,
                ),
                "global_end_index": torch.tensor(
                    [0], dtype=torch.long, device=device, requires_grad=False
                ),
                "local_end_index": torch.tensor(
                    [0], dtype=torch.long, device=device, requires_grad=False
                ),
            }
            for _ in range(self.NUM_TRANSFORMER_BLOCKS)
        ]

    def _initialize_crossattn_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.crossattn_cache = [
            {
                "k": torch.zeros(
                    [batch_size, 512, 12, 128],
                    dtype=dtype, device=device, requires_grad=False,
                ),
                "v": torch.zeros(
                    [batch_size, 512, 12, 128],
                    dtype=dtype, device=device, requires_grad=False,
                ),
                "is_init": False,
            }
            for _ in range(self.NUM_TRANSFORMER_BLOCKS)
        ]

    def _maybe_roll_kv_cache_for_overflow(
        self,
        critic_state: SimpleNamespace,
    ) -> None:
        kv_cache_size: int = critic_state.kv_cache1[0]["k"].size(1)
        kv_cache_end: int = critic_state.kv_cache1[0]["local_end_index"][0].item()
        if kv_cache_size > kv_cache_end:
            return

        critic_state.current_start_frame -= self.num_frame_per_block
        for block_kv in critic_state.kv_cache1:
            b, l, h, d = block_kv["k"].size()
            cache_device: torch.device = block_kv["k"].device
            cache_dtype: torch.dtype = block_kv["k"].dtype
            extension_l: int = int((l / self.local_attn_size) * self.num_frame_per_block)
            k_ext = torch.zeros((b, extension_l, h, d), device=cache_device, dtype=cache_dtype)
            v_ext = torch.zeros((b, extension_l, h, d), device=cache_device, dtype=cache_dtype)
            block_kv["k"] = torch.cat((block_kv["k"], k_ext), dim=1)[:, extension_l:]
            block_kv["v"] = torch.cat((block_kv["v"], v_ext), dim=1)[:, extension_l:]
            block_kv["global_end_index"] -= extension_l
            block_kv["local_end_index"] -= extension_l
