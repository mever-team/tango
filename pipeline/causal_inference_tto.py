"""Causal inference pipeline with white-Gaussian-noise consistency TTO.

This is a focused subset of `causal_inference.py`. It implements only the
path used by the LV-Bench V2V experiment described in
`scripts/tto/v2v/lvbench_test/...lora_r8_bs2.sh` and it intentionally hard-codes
the configuration choices made by that experiment so the rollout is short and
auditable. Concretely, the assumptions are:

    * Loss: `gaussian_forcing` (`WhiteGaussianNoiseConsistencyLoss`).
    * Critic: a frozen copy of the initial generator (`critic_model: init`).
    * Critic type: `single_stage` (no spectral masking of the conditioning).
    * Regularization: `critic_output_mse`.
    * LoRA training (no chaining).
    * `reset_actor_per_block: true` — LoRA is re-initialized at each block.
    * V2V conditioning: `initial_rgb` is required.
    * `warp_denoising_step: true`, `independent_first_frame: false`.

Anything outside this configuration is intentionally not supported here. Use
the original `causal_inference.py` for those experiments.
"""

import time
from types import SimpleNamespace
from typing import Any, List, Optional

import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from tto import trainer


class CausalInferenceTTOPipeline(torch.nn.Module):
    """Causal autoregressive inference with per-step Gaussian-forcing TTO."""

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

        # Memory profiling state — populated per-call in `generate(...)`. When
        # `tto.profile_memory` is set, `_mem(tag)` prints the current+peak GPU
        # allocation at each instrumented site. Default off.
        self._profile_memory: bool = False

        print(f"KV inference with {self.num_frame_per_block} frames per block.")

    def _mem(self, tag: str) -> None:
        if not self._profile_memory:
            return
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        max_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"[mem] {tag:<60}  alloc={alloc:6.2f}  peak={max_alloc:6.2f}  reserved={reserved:6.2f} GB", flush=True)

    # ------------------------------------------------------------------
    # Public entry point.
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def generate(
        self,
        tto_trainer: trainer.Trainer,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_rgb: torch.Tensor,
        return_latents: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, Any]]:
        """Run TTO V2V generation.

        Returns:
            video: decoded RGB video (B, T, C, H, W) in [0, 1] on CPU as float32.
            latents: optimized latent video on CPU (or None if `return_latents`
                is False).
            stats: dict with the following keys:
                wall_clock_seconds (float)
                text_encode_seconds, vae_encode_seconds, rollout_seconds,
                vae_decode_seconds (float; sub-component breakdowns)
                per_step_losses: nested dict
                    {block_index: {step_index: {epoch: {loss_name: float}}}}
        """
        device: torch.device = noise.device
        # Read memory-profile flag from config; reset peak so this call's peak
        # is meaningful in isolation.
        self._profile_memory = bool(tto_trainer.config.tto.get("profile_memory", False))
        if self._profile_memory:
            torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t_start: float = time.perf_counter()
        self._mem("generate() entry")

        # ---- Encode the conditioning video. ----
        torch.cuda.synchronize(device)
        t_vae_enc_start: float = time.perf_counter()
        with torch.no_grad():
            initial_latent: torch.Tensor = self.vae(initial_rgb, mode="encode").to(
                device, dtype=noise.dtype
            )
        torch.cuda.synchronize(device)
        vae_encode_seconds: float = time.perf_counter() - t_vae_enc_start
        self._mem("after VAE encode of initial_rgb")

        batch_size, num_frames, num_channels, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0, (
            "num_frames in noise must be a multiple of num_frame_per_block "
            f"(got {num_frames} % {self.num_frame_per_block})."
        )
        num_blocks: int = num_frames // self.num_frame_per_block
        num_input_frames: int = initial_latent.shape[1]
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

        # ---- Initialize the KV caches and seed them with the conditioning. ----
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
            assert num_input_frames % self.num_frame_per_block == 0, (
                "num_input_frames must be a multiple of num_frame_per_block."
            )
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

            # `tto.optimization_scope` (idea 4) picks between the historical
            # per-sub-timestep TTO and a single per-block trajectory-end TTO.
            optimization_scope: str = str(
                tto_trainer.config.tto.get("optimization_scope", "per_subtimestep")
            )

            # `block_traj_state` carries the per-block TTO outputs (the
            # last-epoch transition noises) into the actual rollout when
            # `optimization_scope == "per_block"`. Stays None otherwise.
            block_traj_state: Optional[SimpleNamespace] = None
            if optimization_scope == "per_block":
                block_traj_state = SimpleNamespace(
                    initial_latent=initial_latent,
                    current_start_frame=current_start_frame,
                    noisy_input=noisy_input,
                    kv_cache1=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                )
                # Run TTO ONCE for this block. The actual 4-step rollout
                # follows in the sub-timestep loop below using the trained
                # LoRA. Losses are recorded under sub-timestep key "block"
                # so the per-step-losses JSON has a stable schema.
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

                # --- Optimize LoRA for this (block, sub-timestep). ---
                # Skipped under `optimization_scope=per_block` (TTO already
                # ran for the whole block above) and under idea 3's
                # `optimize_at_subtimesteps` short-circuit (handled inside
                # `_optimize_block_step`).
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
                    if index < len(self.denoising_step_list) - 1:
                        next_timestep: float = float(self.denoising_step_list[index + 1])
                        # `tto.reuse_critic_noise=true` resolves to a
                        # different source depending on scope:
                        #   * per_subtimestep: `state.last_critic_noise`
                        #     set by `_forward_backward` after the inner
                        #     loop for THIS sub-timestep.
                        #   * per_block: `block_traj_state.last_trajectory_noises[index]`
                        #     set by the LAST TTO epoch's transition tensors.
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

                # Update the main KV cache with a clean-context pass so the
                # next block sees the optimized frames.
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

        # KV caches should be released before the next call (next video).
        self.kv_cache1 = None
        self.crossattn_cache = None

        # ---- Decode the optimized latents back to RGB. ----
        torch.cuda.synchronize(device)
        t_dec_start: float = time.perf_counter()
        with torch.no_grad():
            video: torch.Tensor = self.vae(output.to(device), mode="decode", use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1).float().cpu()
        torch.cuda.synchronize(device)
        vae_decode_seconds: float = time.perf_counter() - t_dec_start

        torch.cuda.synchronize(device)
        wall_clock: float = time.perf_counter() - t_start

        stats: dict[str, Any] = {
            "wall_clock_seconds": wall_clock,
            "text_encode_seconds": text_encode_seconds,
            "vae_encode_seconds": vae_encode_seconds,
            "rollout_seconds": rollout_seconds,
            "vae_decode_seconds": vae_decode_seconds,
            "per_step_losses": per_step_losses,
        }
        latents_out: Optional[torch.Tensor] = output.detach().cpu() if return_latents else None
        return video, latents_out, stats

    # ------------------------------------------------------------------
    # TTO inner loops.
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
        """Run B optimization iterations for one (block, sub-timestep) and
        return the recorded loss values per iteration.

        When `tto.measure_only` is set, no LoRA reset is performed and the
        inner optimization loop is replaced with a single no-grad forward
        whose only purpose is to record what the loss values WOULD be in the
        absence of optimization. The trainable model still has LoRA
        parameters, but `lora_B` is initialized to zero and never updated in
        this mode, so its outputs match the frozen base model bit-for-bit.
        """
        device: torch.device = noise.device
        measure_only: bool = bool(tto_trainer.config.tto.get("measure_only", False))

        # Idea 3 — limit TTO to a subset of the diffusion sub-timesteps.
        # `tto.optimize_at_subtimesteps` is a list of 0-indexed positions
        # within `denoising_step_list`. When set and the current sub-timestep
        # isn't in the list, this method short-circuits: no LoRA reset, no
        # pre-sample, no critic precompute, no inner loop. The outer
        # `generate()` still runs the actual-denoise forward at this
        # sub-timestep so the rollout proceeds normally. Default `None`
        # preserves the historical behaviour (optimise at every sub-timestep).
        optimize_at = tto_trainer.config.tto.get("optimize_at_subtimesteps", None)
        if optimize_at is not None and index not in list(optimize_at):
            return {}

        # LoRA reset policy. The historical default is to reset at every
        # (block, sub-timestep), which is what `reset_actor_per_block=true`
        # has always done in this pipeline despite its name. Setting
        # `reset_actor_per_subtimestep=false` flips to a true per-block
        # reset: the LoRA is reset only at sub-timestep 0 and then evolves
        # across the 4 sub-timesteps of the same block (the 10 epochs at
        # sub-timestep 1 continue refining the LoRA learned at sub-timestep
        # 0, etc.). Skipped in `measure_only` mode: there is nothing to reset.
        if not measure_only and tto_trainer.config.tto.get("reset_actor_per_block", False):
            reset_per_substep: bool = bool(
                tto_trainer.config.tto.get("reset_actor_per_subtimestep", True)
            )
            if reset_per_substep or index == 0:
                tto_trainer.reset_model()

        # Two orthogonal noise-reuse knobs (legacy compatible — same flag
        # names as `pipeline/causal_inference.py`):
        #
        #   * `tto.critic_all_epochs_single_noise=true` pre-samples ONE
        #     noise tensor per (block, sub-timestep) and reuses it across
        #     all 10 inner TTO iterations (`_compute_loss`). Stored on
        #     `state.single_critic_noise`. Variance-reduction of the
        #     critic look-ahead gradient.
        #   * `tto.reuse_critic_noise=true` takes whatever noise was last
        #     consumed by the critic look-ahead and feeds it into the
        #     outer scheduler.add_noise that produces the next
        #     sub-timestep's noisy_input (`generate()` below). Bridges the
        #     TTO optimisation to the actual rollout trajectory.
        #     Tracked on `state.last_critic_noise` by `_forward_backward`.
        #
        # Setting both to true == idea #2: same noise for the 10 epochs
        # AND for generation. Either can be set independently.
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

        # Pre-compute the frozen critic's prediction once for the MSE
        # regularization. Done outside the inner epoch loop because none of
        # its inputs change during optimization.
        self._mem(f"    optimize_block_step({block_index},{index}) entry")
        with torch.no_grad():
            critic_kv: list[dict[str, Any]] = self._clone_cache(state.kv_cache1, device)
            critic_crossattn: list[dict[str, Any]] = self._clone_cache(state.crossattn_cache, device)
            self._mem(f"    after critic_kv+critic_crossattn clone (precompute)")
            _, critic_denoised_pred = tto_trainer.critic_model(
                noisy_image_or_video=state.noisy_input.detach(),
                conditional_dict=conditional_dict,
                timestep=timestep.detach(),
                kv_cache=critic_kv,
                crossattn_cache=critic_crossattn,
                current_start=state.current_start_frame * self.FRAME_SEQ_LENGTH,
            )
            self._mem(f"    after critic_denoised_pred fwd")

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
        """Single no-grad forward + critic look-ahead used in `measure_only`
        mode to record the loss components that would arise during a
        non-optimized rollout."""
        device: torch.device = noise.device
        critic_state = SimpleNamespace(
            noisy_input=state.noisy_input.detach().clone(),
            current_start_frame=state.current_start_frame,
            crossattn_cache=self._clone_cache(state.crossattn_cache, device),
            kv_cache1=self._clone_cache(state.kv_cache1, device),
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
        """One full optimization step (forward + critic look-ahead +
        backward + optimizer step). Returns the per-component loss values."""
        device: torch.device = noise.device

        # Snapshot of caches/inputs that the trainable forward will mutate. The
        # snapshot is discarded after this iteration so the live rollout state
        # is untouched.
        self._mem(f"      forward_backward({block_index},{index},ep={epoch}) entry")
        critic_state = SimpleNamespace(
            noisy_input=state.noisy_input.detach().clone(),
            current_start_frame=state.current_start_frame,
            crossattn_cache=self._clone_cache(state.crossattn_cache, device),
            kv_cache1=self._clone_cache(state.kv_cache1, device),
            # Forward `tto.critic_all_epochs_single_noise`'s pre-sampled
            # tensor (if any) so the critic look-ahead inside `_compute_loss`
            # consumes the exact same noise every epoch.
            shared_next_noise=getattr(state, "single_critic_noise", None),
        )
        critic_timestep: torch.Tensor = timestep.detach().clone()
        self._mem(f"      after per-iteration cache clones")

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
        # the outer `generate()` can reuse it when
        # `tto.reuse_critic_noise=true`. Persisting across iterations gives
        # the legacy "last sampled noise wins" semantics; if
        # `tto.critic_all_epochs_single_noise=true` is also on the value is
        # constant across iterations so the last == the first.
        if used_noise is not None:
            state.last_critic_noise = used_noise.detach()
        self._mem(f"      after _compute_loss")

        tto_trainer.backward_loss(loss)
        self._mem(f"      after backward + optimizer.step()")
        return loss_dict

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
        """Compute the total TTO loss for this (block, sub-timestep, epoch).

        Returns the total loss tensor (with grad) and a dict of detached
        per-component loss values for offline analysis.
        """
        tto_trainer.optimizer.zero_grad(set_to_none=True)

        opt_batch_size: int = tto_trainer.config.tto.get("batch_size", 1)
        device: torch.device = noise.device
        self._mem(f"        _compute_loss entry (B={opt_batch_size})")

        # --- Trainable forward: gradients flow back to LoRA via the critic
        #     look-ahead below. ---
        _, denoised_pred = tto_trainer.model(
            noisy_image_or_video=critic_state.noisy_input,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
            kv_cache=critic_state.kv_cache1,
            crossattn_cache=critic_state.crossattn_cache,
            current_start=critic_state.current_start_frame * self.FRAME_SEQ_LENGTH,
        )
        self._mem(f"        after trainable fwd")

        # MSE regularization against the frozen critic's prediction.
        loss_reg: torch.Tensor = torch.nn.functional.mse_loss(
            denoised_pred, critic_denoised_pred
        )

        # --- Expand caches and conditioning to the optimization batch size B
        #     so the critic look-ahead averages over multiple noise samples. ---
        critic_state.crossattn_cache = self._crossattn_cache_batch_expand(
            critic_state.crossattn_cache, opt_batch_size
        )
        critic_state.kv_cache1 = self._kv_cache_batch_repeat(
            critic_state.kv_cache1, opt_batch_size
        )
        self._mem(f"        after kv_cache batch_repeat (B={opt_batch_size})")
        _, t, c, h, w = denoised_pred.size()
        denoised_pred_b: torch.Tensor = denoised_pred.expand((opt_batch_size, t, c, h, w))
        cond_dict_b: dict[str, torch.Tensor] = {
            "prompt_embeds": conditional_dict["prompt_embeds"].expand(opt_batch_size, -1, -1)
        }

        # --- Build the next-step noisy input for the look-ahead. ---
        # `lookahead_target` controls the critic look-ahead's temporal
        # target:
        #   * `next_subtimestep` (default): existing behaviour. Within-
        #     block look-ahead at `t[index+1]` for `index < last`, next-
        #     block look-ahead with context-noise cache injection at
        #     `index == last`.
        #   * `next_block_no_inject` (Variant 1b): always look one
        #     temporal block ahead at `t[0]`, NO context-noise cache
        #     injection. Same compute cost as the default.
        #   * `next_block_inject` (Variant 1a): always look one temporal
        #     block ahead at `t[0]`, WITH context-noise injection so the
        #     cache state matches actual generation. +1 critic fwd / sub-
        #     timestep (~3-4 GB extra via gradient checkpointing).
        #   * `dual`: BOTH within-block and next-block-no-inject look-
        #     aheads at every sub-timestep where applicable. K/V at the
        #     current block range is snapshotted between the two look-
        #     aheads so the next-block one reads the trainable's K/V.
        #     At the last sub-timestep within-block has no target —
        #     falls through to next-block-no-inject only. Combined as
        #     `wb_w * gauss_wb + nb_w * gauss_nb`.
        # `used_noise` is exported back to the caller so the outer
        # `generate()` can reuse it via `tto.reuse_critic_noise`. Important:
        # the outer rollout runs at `batch_size == noise.shape[0]` (always 1
        # for single-video V2V), while the inner critic look-ahead runs at
        # `opt_batch_size = tto.batch_size`. Export `used_noise` at the
        # OUTER batch dim so the outer `scheduler.add_noise` gets the right
        # shape — match the legacy code at `pipeline/causal_inference.py`
        # which slices `[:1]` before bridging into the next sub-timestep.
        used_noise: Optional[torch.Tensor] = None
        outer_batch: int = int(noise.shape[0])
        lookahead_target: str = str(
            tto_trainer.config.tto.get("lookahead_target", "next_subtimestep")
        )
        is_last_subtimestep: bool = (index >= len(self.denoising_step_list) - 1)

        if lookahead_target == "dual":
            # ---- Dual look-ahead: within-block (if not last) + next-block-no-inject. ----
            wb_weight = float(tto_trainer.config.tto.get("lookahead_within_block_weight", 1.0))
            nb_weight = float(tto_trainer.config.tto.get("lookahead_next_block_weight", 1.0))
            per_moment_loss: dict[str, float] = {}
            gauss_wb: Optional[torch.Tensor] = None
            gauss_nb: torch.Tensor

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

                # Restore K/V so the next-block look-ahead sees the
                # trainable's K/V at the current block range, not the
                # within-block-look-ahead's overwrite.
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
            else:
                gaussian_loss = gauss_nb

            reg_w: float = float(tto_trainer.config.tto.regularization_weight)
            gauss_w: float = float(tto_trainer.config.tto.white_gaussian_noise_weight)
            total_loss: torch.Tensor = gauss_w * gaussian_loss + reg_w * loss_reg
            loss_dict: dict[str, float] = {
                "loss_total": total_loss.detach().cpu().item(),
                "loss_gaussian": gaussian_loss.detach().cpu().item(),
                "loss_reg": loss_reg.detach().cpu().item(),
                "regularization_weight": reg_w,
                "gaussian_weight": gauss_w,
                "critic_timestep": float(nb_next_t),
                "lookahead_target": lookahead_target,
                **per_moment_loss,
            }
            if gauss_wb is not None:
                loss_dict["loss_gaussian_wb"] = gauss_wb.detach().cpu().item()
                loss_dict["loss_gaussian_nb"] = gauss_nb.detach().cpu().item()
                loss_dict["lookahead_within_block_weight"] = wb_weight
                loss_dict["lookahead_next_block_weight"] = nb_weight
            # `reuse_critic_noise` is inert under `dual` — see the
            # corresponding comment in `causal_inference_tto_optimized.py`.
            return total_loss, loss_dict, None

        # ---- Single look-ahead modes. ----
        if lookahead_target == "next_subtimestep" and not is_last_subtimestep:
            # Same block, next sub-timestep: re-noise the trainable prediction.
            critic_next_timestep: float = float(self.denoising_step_list[index + 1])
            # `tto.critic_all_epochs_single_noise` path: a noise tensor was
            # pre-sampled once and threaded in via
            # `critic_state.shared_next_noise`. Reuse it; otherwise sample
            # fresh per iteration (current default).
            shared = getattr(critic_state, "shared_next_noise", None)
            if shared is not None:
                # `denoised_pred_b` has the broadcasted batch axis when
                # `tto.batch_size > 1`; tile the shared single-sample noise
                # to that batch dimension so the per-sample look-ahead sees
                # identical noise across the perturbations.
                if shared.shape[0] != denoised_pred_b.shape[0]:
                    next_noise = shared.expand_as(denoised_pred_b)
                else:
                    next_noise = shared
                # `shared` was sampled at outer batch dim already.
                used_noise = shared
            else:
                next_noise = torch.randn_like(denoised_pred_b)
                # Take the first opt_batch_size slice down to the outer
                # batch dim; clone so subsequent ops on next_noise (none
                # here, but defensive) can't perturb the stored copy.
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
            # Next-block look-ahead. Inject context-noise into the cache
            # in two cases:
            #   * `next_subtimestep` at last sub-timestep (existing
            #     behaviour — preserves rollout-faithful cache state).
            #   * `next_block_inject` at any sub-timestep.
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
            # NOTE: This mirrors the original behaviour: the slice picks the
            # current block's noise as a stand-in for the "next block" pure
            # noise. Near the rollout end this also avoids reading past the
            # end of the noise tensor.
            critic_state.noisy_input = noise[
                :,
                critic_state.current_start_frame - num_input_frames - current_num_frames:
                critic_state.current_start_frame - num_input_frames,
            ].expand((opt_batch_size, t, c, h, w))

            # When the KV cache is already at capacity, slide it forward by
            # one block so gradient checkpointing during backward sees a
            # consistent token range.
            self._maybe_roll_kv_cache_for_overflow(critic_state)

        # --- Critic look-ahead forward (frozen; gradients pass through inputs). ---
        critic_timestep_next: torch.Tensor = torch.full(
            [opt_batch_size, current_num_frames],
            critic_next_timestep,
            device=device,
            dtype=torch.float32,
        )
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

        # --- White-Gaussian-noise consistency loss + total. ---
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
        # `used_noise` is the noise tensor that fed the critic look-ahead
        # this iteration (None on the last sub-timestep, where the
        # look-ahead branches to a context-noise re-noise instead).
        # `_forward_backward` writes this onto the outer `state` so the
        # last iteration's noise can be reused by the outer
        # `scheduler.add_noise` when `tto.reuse_critic_noise=true`.
        return total_loss, loss_dict, used_noise

    # ------------------------------------------------------------------
    # Idea 4 — per-block (trajectory-end) TTO.
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
        """Idea 4: single TTO optimisation spanning all 4 sub-timesteps of
        a block. Each epoch runs the trainable through the full 4-step
        denoising trajectory (an autograd-traced chain), then a critic
        look-ahead at the end. Loss = `gauss_w * gaussian_forcing(pred_noise)
        + reg_w * MSE(trainable_x0_final, critic_x0_final)` where
        `critic_x0_final` is pre-computed once from the frozen critic's
        own 4-step trajectory.

        Noise reuse honours the same two flags as the per-sub-timestep
        mode: `critic_all_epochs_single_noise` pre-samples the 3
        transition noises and pins them across epochs;
        `reuse_critic_noise` bridges the LAST epoch's transition noises
        to the actual rollout that runs after this method returns."""
        device: torch.device = noise.device
        measure_only: bool = bool(tto_trainer.config.tto.get("measure_only", False))
        num_subtimesteps: int = len(self.denoising_step_list)
        num_transitions: int = num_subtimesteps - 1

        # LoRA reset is naturally per-block in this mode. The
        # `reset_actor_per_subtimestep` knob is meaningless here (there
        # are no sub-timestep boundaries inside the optimisation), so
        # `reset_actor_per_block=true` is the only relevant flag.
        if not measure_only and tto_trainer.config.tto.get("reset_actor_per_block", False):
            tto_trainer.reset_model()

        # Pre-sample the 3 transition noises (input to scheduler.add_noise
        # between consecutive sub-timesteps) when
        # `critic_all_epochs_single_noise=true`. Same tensor list is then
        # consumed by every TTO epoch AND by the frozen-critic precompute
        # AND (if `reuse_critic_noise=true`) by the actual rollout —
        # binding the optimisation to the exact trajectory we'll render.
        use_single_noise: bool = bool(
            tto_trainer.config.tto.get("critic_all_epochs_single_noise", False)
        )
        state.single_trajectory_noises: Optional[list[torch.Tensor]] = None
        state.last_trajectory_noises: Optional[list[torch.Tensor]] = None
        if use_single_noise and num_transitions > 0:
            state.single_trajectory_noises = [
                torch.randn(
                    [batch_size, current_num_frames,
                     noise.shape[2], noise.shape[3], noise.shape[4]],
                    device=device, dtype=noise.dtype,
                )
                for _ in range(num_transitions)
            ]

        # --- Frozen critic precompute: run the critic through the same 4
        #     sub-timesteps so we have `critic_x0_final` as the MSE-reg
        #     target. Uses the pre-sampled transition noises when
        #     `single_trajectory_noises` is set; otherwise samples its own
        #     fresh noises (which will NOT match the trainable's per-epoch
        #     transition noises — but that's fine, the reg target only
        #     depends on the FINAL x0). ---
        self._mem(f"    optimize_block_trajectory({block_index}) entry")
        with torch.no_grad():
            critic_kv: list[dict[str, Any]] = self._clone_cache(state.kv_cache1, device)
            critic_crossattn: list[dict[str, Any]] = self._clone_cache(state.crossattn_cache, device)
            self._mem(f"    after critic precompute clones")
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
                t_val_f = float(t_val)
                t_tensor = torch.full(
                    [batch_size, current_num_frames], t_val_f,
                    device=device, dtype=torch.float32,
                )
                _, critic_x0 = tto_trainer.critic_model(
                    noisy_image_or_video=critic_noisy,
                    conditional_dict=conditional_dict,
                    timestep=t_tensor,
                    kv_cache=critic_kv,
                    crossattn_cache=critic_crossattn,
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
            self._mem(f"    after critic trajectory precompute")
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
                # On the LAST epoch, record the transition noises onto
                # `state.last_trajectory_noises` so the outer rollout can
                # bridge them when `reuse_critic_noise=true`.
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
        """One full-trajectory TTO iteration: 4 chained trainable forwards
        + end-of-trajectory critic look-ahead + loss assembly. Returns the
        backward-able loss tensor + a detached `loss_dict`. Note this
        method does NOT call `optimizer.zero_grad()` / `backward()` — the
        caller (`_optimize_block_trajectory`) wraps it so the no-grad
        measure-only path can also reuse it."""
        device: torch.device = noise.device
        num_subtimesteps: int = len(self.denoising_step_list)
        num_transitions: int = num_subtimesteps - 1

        tto_trainer.optimizer.zero_grad(set_to_none=True)

        # Clone the live caches so the trainable trajectory can mutate them
        # freely without affecting the outer rollout.
        critic_state = SimpleNamespace(
            noisy_input=state.noisy_input.detach().clone(),
            current_start_frame=state.current_start_frame,
            crossattn_cache=self._clone_cache(state.crossattn_cache, device),
            kv_cache1=self._clone_cache(state.kv_cache1, device),
        )
        self._mem(f"      after per-iteration cache clones")

        # Transition noises this epoch.
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

        # --- Run the trainable through all 4 sub-timesteps. ---
        noisy_input: torch.Tensor = critic_state.noisy_input
        trainable_x0_final: Optional[torch.Tensor] = None
        for idx, t_val in enumerate(self.denoising_step_list):
            t_val_f = float(t_val)
            t_tensor = torch.full(
                [batch_size, current_num_frames], t_val_f,
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

        # --- MSE reg against the frozen critic's final x0. ---
        loss_reg: torch.Tensor = torch.nn.functional.mse_loss(
            trainable_x0_final, critic_x0_final
        )

        # --- End-of-trajectory critic look-ahead, mirroring `_compute_loss`'s
        #     last-sub-timestep branch: clean-context fwd, advance frame,
        #     read next chunk's noise, critic look-ahead. ---
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

        # On the final epoch, persist the transition noises so the outer
        # `generate()` actual-rollout can reuse them when
        # `reuse_critic_noise=true`. Detach to release the autograd graph.
        if record_last_noises:
            state.last_trajectory_noises = [n.detach() for n in transition_noises]

        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # KV-cache helpers (single-cache variants of the original utilities).
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

    def _clone_cache(
        self,
        cache: list[dict[str, Any]],
        device: torch.device,
    ) -> list[dict[str, Any]]:
        return [
            {
                k: (v.detach().clone().to(device).requires_grad_(False)
                    if isinstance(v, torch.Tensor) else v)
                for k, v in cache[i].items()
            }
            for i in range(self.NUM_TRANSFORMER_BLOCKS)
        ]

    def _snapshot_kv_at_last_block(
        self,
        kv_cache: list[dict[str, torch.Tensor]],
        current_num_frames: int,
    ) -> dict:
        """Snapshot the K/V slice at the MOST RECENTLY WRITTEN block, i.e.
        `[local_end_index - current_num_frames * FRAME_SEQ_LENGTH :
        local_end_index)` per layer.

        Used by `lookahead_target=dual`: the trainable forward writes K/V
        at the current block range; a within-block critic look-ahead
        overwrites that same slice in-place; we then restore the snapshot
        so a subsequent next-block critic look-ahead reads the trainable's
        (not the within-block-look-ahead's) K/V. Cost is roughly
        `current_num_frames * 1560 * 12 * 128 * 4 * 30` ≈ 85 MB.
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
        cache: list[dict[str, Any]],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        for entry in cache:
            entry["k"] = entry["k"].repeat((batch_size, 1, 1, 1))
            entry["v"] = entry["v"].repeat((batch_size, 1, 1, 1))
            entry["global_end_index"] = entry["global_end_index"].repeat(batch_size)
            entry["local_end_index"] = entry["local_end_index"].repeat(batch_size)
        return cache

    def _maybe_roll_kv_cache_for_overflow(
        self,
        critic_state: SimpleNamespace,
    ) -> None:
        """Slide the KV cache forward by one block when it would overflow.

        This is required so gradient checkpointing during the backward pass
        can re-attend to a consistent token range. Without it, the cache would
        be silently overwritten and the autograd recomputation would diverge
        from the original forward.
        """
        kv_cache_size: int = critic_state.kv_cache1[0]["k"].size(1)
        kv_cache_end: int = critic_state.kv_cache1[0]["local_end_index"][0].item()
        if kv_cache_size > kv_cache_end:
            return  # plenty of room left

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
