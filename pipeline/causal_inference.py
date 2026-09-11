import functools
from types import SimpleNamespace
from typing import List, Optional, Any

import einops
import torch
import box
import torch.distributed as dist

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
# from utils.distributed import fsdp_wrap
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from tto.masking import masking_strategies
from tto import losses, trainer


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae
        # if self.vae is not None and dist.is_initialized():
        #     self.vae = fsdp_wrap(
        #         self.vae,
        #         sharding_strategy=args.sharding_strategy,
        #         mixed_precision=args.mixed_precision,
        #         wrap_strategy=args.generator_fsdp_wrap_strategy,
        #         cpu_offload=True,
        #         min_num_params=int(5e6)
        #     )

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video


    def inference_tto(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_rgb: torch.Tensor,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]] | torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        # Perform low and high pass filtering on the initial RGB frames.
        masking_radius: int = 16  # TODO: Make it variable.
        masked_initial_rgb: dict[str, torch.Tensor] = masking_strategies.spectral_mask_sequence(
            einops.rearrange(initial_rgb, "b c t h w -> b t c h w").float(),
            masking_radius=masking_radius
        )
        masked_initial_rgb = {k: einops.rearrange(
                                v, "b t c h w -> b c t h w").to(dtype=initial_rgb.dtype)
                              for k, v in masked_initial_rgb.items()}
        initial_rgb: dict[str, torch.Tensor] = {"original": initial_rgb}
        initial_rgb.update(masked_initial_rgb)

        # Encode RGB to latent.
        initial_latents: dict[str, torch.Tensor] = {
            k: self.vae.encode_to_latent(v).to(noise.device, dtype=noise.dtype)
            for k, v in initial_rgb.items()
        }
        del initial_rgb

        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latents is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latents["original"].shape[1] if initial_latents is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV caches to all zeros.
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                num_caches=len(initial_latents)
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                num_caches=len(initial_latents)
            )
        else:
            # reset cross attn caches
            for block_index in range(self.num_transformer_blocks):
                assert isinstance(self.crossattn_cache, list)
                for crossattn_cache in self.crossattn_cache:
                    crossattn_cache[block_index]["is_init"] = False
            # reset kv caches
            for kv_cache1 in self.kv_cache1:
                for block_index in range(len(kv_cache1)):
                    kv_cache1[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    kv_cache1[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)

        # Create a separate state for each of the initial latents.
        latent_states: dict[str, box.Box] = {n: box.Box({
            "initial_latent": initial_latent,
            "current_start_frame": 0,
            "kv_cache1": self.kv_cache1[i],
            "crossattn_cache": self.crossattn_cache[i],
            "output": torch.zeros(
                [batch_size, num_output_frames, num_channels, height, width],
                device=noise.device,
                dtype=noise.dtype
            )
        }) for i, (n, initial_latent) in enumerate(initial_latents.items())}

        # Step 2: Cache context features
        for n, s in latent_states.items():
            if s.initial_latent is not None:
                timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
                if self.independent_first_frame:
                    # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                    assert (num_input_frames - 1) % self.num_frame_per_block == 0
                    num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                    s.output[:, :1] = s.initial_latent[:, :1]
                    self.generator(
                        noisy_image_or_video=s.initial_latent[:, :1],
                        conditional_dict=conditional_dict,
                        timestep=timestep * 0,
                        kv_cache=s.kv_cache1,
                        crossattn_cache=s.crossattn_cache,
                        current_start=s.current_start_frame * self.frame_seq_length,
                    )
                    s.current_start_frame += 1
                else:
                    # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                    assert num_input_frames % self.num_frame_per_block == 0
                    num_input_blocks = num_input_frames // self.num_frame_per_block

                for _ in range(num_input_blocks):
                    current_ref_latents = \
                        s.initial_latent[:, s.current_start_frame:s.current_start_frame + self.num_frame_per_block]
                    s.output[:, s.current_start_frame:s.current_start_frame + self.num_frame_per_block] = current_ref_latents
                    self.generator(
                        noisy_image_or_video=current_ref_latents,
                        conditional_dict=conditional_dict,
                        timestep=timestep * 0,
                        kv_cache=s.kv_cache1,
                        crossattn_cache=s.crossattn_cache,
                        current_start=s.current_start_frame * self.frame_seq_length,
                    )
                    s.current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        all_tto_losses: dict[str, torch.Tensor] = {}

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latents is None:
            all_num_frames = [1] + all_num_frames

        del initial_latents
        for n, s in latent_states.items():
            del s.initial_latent

        for block_index, current_num_frames in enumerate(all_num_frames):
            if profile:
                block_start.record()

            for n, s in latent_states.items():
                s.noisy_input = noise[
                    :, s.current_start_frame - num_input_frames:s.current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                for n, s in latent_states.items():
                    if index < len(self.denoising_step_list) - 1:
                        _, s.denoised_pred = self.generator(
                            noisy_image_or_video=s.noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=s.kv_cache1,
                            crossattn_cache=s.crossattn_cache,
                            current_start=s.current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        s.noisy_input = self.scheduler.add_noise(
                            s.denoised_pred.flatten(0, 1),
                            torch.randn_like(s.denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, s.denoised_pred.shape[:2])
                    else:
                        # for getting real output
                        _, s.denoised_pred = self.generator(
                            noisy_image_or_video=s.noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=s.kv_cache1,
                            crossattn_cache=s.crossattn_cache,
                            current_start=s.current_start_frame * self.frame_seq_length
                        )
                # for n, s in latent_states.items():
                #     # for item in s.kv_cache1:
                #     #     for k, v in item.items():
                #     #         item[k] = v.cpu()
                #     # for item in s.crossattn_cache:
                #     #     for k, v in item.items():
                #     #         if isinstance(v, torch.Tensor):
                #     #             item[k] = v.cpu()
                #     s.output = s.output.cpu()
                #     s.noisy_input = s.noisy_input.cpu()

                tto_losses: dict[str, torch.Tensor] = losses.compute_spectral_masked_similarities(
                    {n: s.denoised_pred for n, s in latent_states.items()},
                    masking_radius,
                    self.vae.decode_to_pixel,
                    self.vae.encode_to_latent,
                    f"block_{block_index}_t_{current_timestep}",
                )
                all_tto_losses.update(tto_losses)

                # for n, s in latent_states.items():
                #     # for item in s.kv_cache1:
                #     #     for k, v in item.items():
                #     #         item[k] = v.cuda()
                #     # for item in s.crossattn_cache:
                #     #     for k, v in item.items():
                #     #         if isinstance(v, torch.Tensor):
                #     #             item[k] = v.cuda()
                #     s.output = s.output.cuda()
                #     s.noisy_input = s.noisy_input.cuda()

            # Step 3.2: record the model's output
            for n, s in latent_states.items():
                s.output[:, s.current_start_frame:s.current_start_frame + current_num_frames] = s.denoised_pred

                # Step 3.3: rerun with timestep zero to update KV cache using clean context
                context_timestep = torch.ones_like(timestep) * self.args.context_noise
                self.generator(
                    noisy_image_or_video=s.denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=s.kv_cache1,
                    crossattn_cache=s.crossattn_cache,
                    current_start=s.current_start_frame * self.frame_seq_length,
                )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            for n, s in latent_states.items():
                s.current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Free-up a bit of memory.
        for n, s in latent_states.items():
            del s.kv_cache1
            del s.crossattn_cache
            del s.denoised_pred
            del s.noisy_input
        self.kv_cache1 = None
        self.crossattn_cache = None

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(latent_states["original"].output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return (video, latent_states["original"].output,
                    {k: v.item() for k, v in all_tto_losses.items()})
        else:
            return video

    @torch.enable_grad()
    def inference_train_tto(
        self,
        tto_trainer: trainer.Trainer,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_rgb: torch.Tensor | None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, Any]]] | torch.Tensor:
        """
        Perform inference on the given noise and text prompts using test-time optimization.

        Inputs:
            tto_trainer:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        with torch.no_grad():
            masked_initial_rgb: Optional[dict[str, torch.Tensor]] = None
            if tto_trainer.config.tto.get("critic_type", None) != "single_stage":
                # Perform low and high pass filtering on the initial RGB frames.
                masking_radius: int = tto_trainer.masking_radius
                masked_initial_rgb = masking_strategies.spectral_mask_sequence(
                    einops.rearrange(initial_rgb, "b c t h w -> b t c h w").float(),
                    masking_radius=masking_radius
                )
                masked_initial_rgb = {k: einops.rearrange(
                                        v, "b t c h w -> b c t h w").to(dtype=initial_rgb.dtype)
                                      for k, v in masked_initial_rgb.items()}
            initial_rgb: dict[str, torch.Tensor | None] = {"original": initial_rgb}
            if masked_initial_rgb is not None:
                initial_rgb.update(masked_initial_rgb)
                del masked_initial_rgb

            # Encode RGB to latent.
            initial_latents: dict[str, torch.Tensor | None] = {
                k: (self.vae(v, mode="encode") if v.size(dim=2) > 1 else self.vae.encode_to_latent(v)).to(noise.device, dtype=noise.dtype) if v is not None else None
                for k, v in initial_rgb.items()
            }
            del initial_rgb
            torch.cuda.empty_cache()

            batch_size, num_frames, num_channels, height, width = noise.shape
            if not self.independent_first_frame or (self.independent_first_frame and initial_latents is not None):
                # If the first frame is independent and the first frame is provided, then the number of frames in the
                # noise should still be a multiple of num_frame_per_block
                assert num_frames % self.num_frame_per_block == 0
                num_blocks = num_frames // self.num_frame_per_block
            else:
                # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
                assert (num_frames - 1) % self.num_frame_per_block == 0
                num_blocks = (num_frames - 1) // self.num_frame_per_block
            num_input_frames = initial_latents["original"].shape[1] if initial_latents["original"] is not None else 0
            num_output_frames = num_frames + num_input_frames  # add the initial latent frames
            self.text_encoder.to(tto_trainer.device)
            conditional_dict = self.text_encoder(
                text_prompts=text_prompts
            )
            self.text_encoder.to("cpu")
        # if low_memory:
        #     gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
        #     move_model_to_device_with_memory_preservation(
        #         self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
        #     )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Tracks the spatial similarity of the blocks generated by the unoptimized model
        # with the conditional block.
        unoptimized_spatial_similarities: dict[int, dict[int, torch.Tensor]] = {}
        # Tracks the temporal similarity of the blocks
        # unoptimized_temporal_similarities: dict[int, dict[int, torch.Tensor]] = {}

        # Store the denoised latents at the end of each epoch for later returning all of them.
        videos_per_epoch: dict[str, dict[str, Any]] = {}

        # First epoch is used just to compute original model's metrics.
        # Last epoch is used to compute
        for tto_epoch in range(tto_trainer.tto_epochs+2):
            if tto_trainer.loss == "gaussian_forcing" and (tto_epoch < tto_trainer.tto_epochs+1):
                # In gaussian forcing, actual epochs are being performed per step.
                continue
            with torch.no_grad():
                # Step 1: Initialize KV caches to all zeros.
                if self.kv_cache1 is None:
                    self._initialize_kv_cache(
                        batch_size=batch_size,
                        dtype=noise.dtype,
                        device=noise.device,
                        num_caches=len(initial_latents)
                    )
                    self._initialize_crossattn_cache(
                        batch_size=batch_size,
                        dtype=noise.dtype,
                        device=noise.device,
                        num_caches=len(initial_latents)
                    )
                else:
                    if len(initial_latents) > 1:
                        # reset cross attn caches
                        for block_index in range(self.num_transformer_blocks):
                            assert isinstance(self.crossattn_cache, list)
                            for crossattn_cache in self.crossattn_cache:
                                crossattn_cache[block_index]["is_init"] = False
                        # reset kv caches
                        for kv_cache1 in self.kv_cache1:
                            for block_index in range(len(kv_cache1)):
                                kv_cache1[block_index]["global_end_index"] = torch.tensor(
                                    [0], dtype=torch.long, device=noise.device)
                                kv_cache1[block_index]["local_end_index"] = torch.tensor(
                                    [0], dtype=torch.long, device=noise.device)
                    else:
                        # reset cross attn caches
                        for block_index in range(self.num_transformer_blocks):
                            assert isinstance(self.crossattn_cache, list)
                            self.crossattn_cache[block_index]["is_init"] = False
                        # reset kv caches
                        for block_index in range(len(self.kv_cache1)):
                            self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                                [0], dtype=torch.long, device=noise.device)
                            self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                                [0], dtype=torch.long, device=noise.device)

                # Create a separate state for each of the initial latents.
                latent_states: dict[str, SimpleNamespace] = {n: SimpleNamespace(
                    initial_latent = initial_latent,
                    current_start_frame = 0,
                    kv_cache1 = self.kv_cache1[i] if len(initial_latents) > 1 else self.kv_cache1,
                    crossattn_cache = self.crossattn_cache[i] if len(initial_latents) > 1 else self.crossattn_cache,
                    output = torch.zeros(
                        [batch_size, num_output_frames, num_channels, height, width],
                        device=noise.device,
                        dtype=noise.dtype
                    )
                ) for i, (n, initial_latent) in enumerate(initial_latents.items())}

                # Step 2: Cache context features
                for n, s in latent_states.items():
                    if s.initial_latent is not None:
                        timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
                        if self.independent_first_frame:
                            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                            assert (num_input_frames - 1) % self.num_frame_per_block == 0
                            num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                            s.output[:, :1] = s.initial_latent[:, :1]
                            tto_trainer.model(
                                noisy_image_or_video=s.initial_latent[:, :1],
                                conditional_dict=conditional_dict,
                                timestep=timestep * 0,
                                kv_cache=s.kv_cache1,
                                crossattn_cache=s.crossattn_cache,
                                current_start=s.current_start_frame * self.frame_seq_length,
                            )
                            s.current_start_frame += 1
                        else:
                            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                            assert num_input_frames % self.num_frame_per_block == 0
                            num_input_blocks = num_input_frames // self.num_frame_per_block

                        for _ in range(num_input_blocks):
                            current_ref_latents = \
                                s.initial_latent[:, s.current_start_frame:s.current_start_frame + self.num_frame_per_block]
                            s.output[
                                :, s.current_start_frame:s.current_start_frame + self.num_frame_per_block] = current_ref_latents
                            tto_trainer.model(
                                noisy_image_or_video=current_ref_latents,
                                conditional_dict=conditional_dict,
                                timestep=timestep * 0,
                                kv_cache=s.kv_cache1,
                                crossattn_cache=s.crossattn_cache,
                                current_start=s.current_start_frame * self.frame_seq_length,
                            )
                            s.current_start_frame += self.num_frame_per_block

            if profile:
                init_end.record()
                torch.cuda.synchronize()
                diffusion_start.record()

            # Step 3: Temporal denoising loop
            all_num_frames = [self.num_frame_per_block] * num_blocks
            if self.independent_first_frame and initial_latents is None:
                all_num_frames = [1] + all_num_frames

            for block_index, current_num_frames in enumerate(all_num_frames):
                if profile:
                    block_start.record()

                for n, s in latent_states.items():
                    s.noisy_input = noise[
                        :, s.current_start_frame - num_input_frames:s.current_start_frame + current_num_frames - num_input_frames]

                # Step 3.1: Spatial denoising loop
                for index, current_timestep in enumerate(self.denoising_step_list):
                    print(f"current_timestep: {current_timestep}")
                    # set current timestep
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep
                    # torch.cuda.memory._record_memory_history(max_entries=100000)
                    sampled_next_step_noise: torch.Tensor | None = None
                    if (0 < tto_epoch < tto_trainer.tto_epochs+1
                            or (tto_trainer.loss == "gaussian_forcing" and tto_epoch == tto_trainer.tto_epochs+1)):  # 0 epoch is init, final epoch is actual generation
                        if "crossattn_cache" in locals():
                            del crossattn_cache
                        if "kv_cache1" in locals():
                            del kv_cache1
                        if "initial_latent" in locals():
                            del initial_latents
                        sampled_next_step_noise = self.tto_optimize(
                            tto_trainer,
                            latent_states,
                            timestep,
                            conditional_dict,
                            index,
                            block_index,
                            batch_size,
                            current_num_frames,
                            noise,
                            num_input_frames,
                            unoptimized_spatial_similarities,
                            tto_epoch
                        )
                        sampled_next_step_noise = sampled_next_step_noise[:1]
                    if (index == 0 and tto_trainer.config.tto.get("reuse_critic_noise", False)
                            and sampled_next_step_noise is not None):
                        for n, s in latent_states.items():
                            s.noisy_input = sampled_next_step_noise
                    # torch.cuda.memory._dump_snapshot(f"memory_snapshot_tto_{tto_epoch}_{block_index}_{index}.pickle")
                    # torch.cuda.memory._record_memory_history(enabled=None)

                    # On initial epoch compute the required metrics with the base model. On TTO training epochs
                    # use the updated model to recompute the input to the next timestep. On the final epoch
                    # compute the final video entirely using the updated model.
                    with torch.no_grad():
                        next_timestep_noise: Optional[torch.Tensor] = None
                        for n, s in latent_states.items():
                            if index < len(self.denoising_step_list) - 1:
                                _, s.denoised_pred = tto_trainer.model(
                                    noisy_image_or_video=s.noisy_input,
                                    conditional_dict=conditional_dict,
                                    timestep=timestep,
                                    kv_cache=s.kv_cache1,
                                    crossattn_cache=s.crossattn_cache,
                                    current_start=s.current_start_frame * self.frame_seq_length
                                )

                                next_timestep = self.denoising_step_list[index + 1]
                                if (tto_trainer.config.tto.get("perturbations_noise", None) != "same"
                                        or next_timestep_noise is None):
                                    if (tto_trainer.config.tto.get("reuse_critic_noise", False)
                                            and sampled_next_step_noise is not None):
                                        next_timestep_noise = sampled_next_step_noise
                                    else:
                                        # When the same noise corruption should be used for all the perturbed inputs,
                                        # generate noise only once per timestep and use it on all inputs.
                                        next_timestep_noise = torch.randn_like(s.denoised_pred)

                                s.noisy_input = self.scheduler.add_noise(
                                    s.denoised_pred.detach().flatten(0, 1),
                                    next_timestep_noise.flatten(0, 1),
                                    next_timestep * torch.ones(
                                        [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                                ).unflatten(0, s.denoised_pred.shape[:2])
                                if dist.is_initialized() and dist.get_world_size() > 1:
                                    dist.broadcast(s.noisy_input, src=0)
                            else:
                                # for getting real output
                                _, s.denoised_pred = tto_trainer.model(
                                    noisy_image_or_video=s.noisy_input,
                                    conditional_dict=conditional_dict,
                                    timestep=timestep,
                                    kv_cache=s.kv_cache1,
                                    crossattn_cache=s.crossattn_cache,
                                    current_start=s.current_start_frame * self.frame_seq_length
                                )
                        if latent_states["original"].initial_latent is not None and tto_epoch == 0:
                            # Compute spatial similarities with the original model. No weights update.
                            unoptimized_spatial_similarities[block_index] = unoptimized_spatial_similarities.get(
                                block_index, {}
                            )
                            unoptimized_spatial_similarities[block_index][index] = losses.compute_spatial_similarity(
                                latent_states["original"].denoised_pred, latent_states["original"].initial_latent
                            ).detach()
                            torch.cuda.empty_cache()

                # Step 3.2: record the model's output
                with torch.no_grad():
                    for n, s in latent_states.items():
                        s.output[:, s.current_start_frame:s.current_start_frame + current_num_frames] = s.denoised_pred

                        # Step 3.3: rerun with timestep zero to update KV cache using clean context
                        context_timestep = torch.ones_like(timestep) * self.args.context_noise
                        tto_trainer.model(
                            noisy_image_or_video=s.denoised_pred,
                            conditional_dict=conditional_dict,
                            timestep=context_timestep,
                            kv_cache=s.kv_cache1,
                            crossattn_cache=s.crossattn_cache,
                            current_start=s.current_start_frame * self.frame_seq_length,
                        )

                if profile:
                    block_end.record()
                    torch.cuda.synchronize()
                    block_time = block_start.elapsed_time(block_end)
                    block_times.append(block_time)

                # Step 3.4: update the start and end frame indices
                for n, s in latent_states.items():
                    s.current_start_frame += current_num_frames

            if tto_epoch == 0:
                epoch_label: str = "init"
            elif tto_epoch == tto_trainer.tto_epochs+1:
                epoch_label: str = "final"
            else:
                epoch_label: str = str(tto_epoch)
            for n, s in latent_states.items():
                videos_per_epoch[f"{n}_tto_epoch_{epoch_label}"] = {
                    "video": s.output.detach().cpu(),
                    "tto_epoch": epoch_label,
                    "label": n
                }

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Free-up a bit of memory.
        for n, s in latent_states.items():
            del s.kv_cache1
            del s.crossattn_cache
            del s.denoised_pred
            del s.noisy_input
        self.kv_cache1 = None
        self.crossattn_cache = None

        # Step 4: Decode the outputs.
        with torch.no_grad():
            for label, video_data in videos_per_epoch.items():
                video = self.vae(video_data["video"].to(tto_trainer.device), mode="decode", use_cache=False)
                video = (video * 0.5 + 0.5).clamp(0, 1)
                videos_per_epoch[label]["video"] = video.float().cpu()
        # For convenience, also return separately the final video.
        video = videos_per_epoch["original_tto_epoch_final"]["video"]

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, latent_states["original"].output, videos_per_epoch
        else:
            return video

    def tto_optimize(
        self,
        tto_trainer,
        latent_states,
        timestep,
        conditional_dict,
        index,
        block_index,
        batch_size,
        current_num_frames,
        noise,
        num_input_frames,
        unoptimized_spatial_similarities,
        tto_epoch,
        offload_main_latent_state: bool = False
    ) -> torch.Tensor:
        original_device: torch.device = noise.device
        if offload_main_latent_state:
            # Move the original latent state to CPU to save memory.
            for i, s in enumerate(latent_states.values()):
                s.initial_latent = s.initial_latent.to("cpu") if s.initial_latent is not None else None
                s.noisy_input = s.noisy_input.to("cpu")
                s.output = s.output.to("cpu")
                if hasattr(s, "denoised_pred"):
                    s.denoised_pred = s.denoised_pred.to("cpu")
                self._move_kv_cache_to(s.kv_cache1, torch.device("cpu"))
                self.kv_cache1[i] = s.kv_cache1
                self._move_crossattn_cache_to(s.crossattn_cache, torch.device("cpu"))
                self.crossattn_cache[i] = s.crossattn_cache
            torch.cuda.empty_cache()

        if tto_trainer.config.tto.get("reset_actor_per_block", None):
            tto_trainer.reset_model()
            print("Actor reset.")

        if (tto_trainer.use_lora and tto_trainer.lora_chain_per_n_blocks
                and block_index > 0 and (block_index % tto_trainer.lora_chain_per_n_blocks == 0) and index == 0):
            tto_trainer.chain_lora()
            print("Lora chain.")

        # If regularization with the output of the critic is required, compute it here once using
        # a new copy of the caches and pass it to all the optimization epochs.
        critic_denoised_pred: torch.Tensor | None = None
        if tto_trainer.config.tto.get("regularization", None) == "critic_output_mse":
            with torch.no_grad():
                reg_critic_latent_states: dict[str, SimpleNamespace] = {
                    n: SimpleNamespace(
                        noisy_input=s.noisy_input.detach().clone().to(original_device),
                        current_start_frame=s.current_start_frame,
                        initial_latent=s.initial_latent.detach().clone().to(original_device) if s.initial_latent is not None else None,
                        crossattn_cache=self._clone_cache(s.crossattn_cache, original_device),
                        kv_cache1=self._clone_cache(s.kv_cache1, original_device)
                    )
                    for n, s in latent_states.items()}
                reg_critic_timestep: torch.Tensor = timestep.detach().clone().to(original_device)
                for n, s in reg_critic_latent_states.items():
                    _, s.denoised_pred = tto_trainer.critic_model(
                        noisy_image_or_video=s.noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=reg_critic_timestep,
                        kv_cache=s.kv_cache1,
                        crossattn_cache=s.crossattn_cache,
                        current_start=s.current_start_frame * self.frame_seq_length
                    )
                critic_denoised_pred = reg_critic_latent_states["original"].denoised_pred
                del reg_critic_latent_states
                del reg_critic_timestep

        sampled_critic_noise: torch.Tensor | None = None
        if tto_trainer.loss == "gaussian_forcing":
            # enable_checkpoint_debug: bool = block_index >= 7 or (block_index == 6 and index == 3 and torch.all(timestep <= 650.).item())
            # with torch.utils.checkpoint.set_checkpoint_debug_enabled(enable_checkpoint_debug):
            tto_epochs: int = tto_trainer.tto_epochs
            presampled_noise: torch.Tensor | None = None
            for tto_epoch in range(tto_epochs):
                sampled_critic_noise = self.tto_forward_backward(
                    tto_trainer,
                    latent_states,
                    timestep,
                    conditional_dict,
                    index,
                    block_index,
                    batch_size,
                    current_num_frames,
                    noise,
                    num_input_frames,
                    unoptimized_spatial_similarities,
                    tto_epoch,
                    original_device,
                    critic_denoised_pred,
                    presampled_noise=presampled_noise
                )

                if tto_trainer.config.tto.get("critic_all_epochs_single_noise", False):
                    presampled_noise = sampled_critic_noise
        else:
            self.tto_forward_backward(
                tto_trainer,
                latent_states,
                timestep,
                conditional_dict,
                index,
                block_index,
                batch_size,
                current_num_frames,
                noise,
                num_input_frames,
                unoptimized_spatial_similarities,
                tto_epoch,
                original_device
            )

        # torch.cuda.empty_cache()

        if offload_main_latent_state:
            # Move the original latent state back to GPU.
            for s in latent_states.values():
                s.initial_latent = s.initial_latent.to(original_device) if s.initial_latent is not None else None
                s.noisy_input = s.noisy_input.to(original_device)
                s.output = s.output.to(original_device)
                if hasattr(s, "denoised_pred"):
                    s.denoised_pred = s.denoised_pred.to(original_device)
                self._move_kv_cache_to(s.kv_cache1, original_device)
                self._move_crossattn_cache_to(s.crossattn_cache, original_device)

        # for s in latent_states.values():
        #     self._detach_a_kv_cache(s.kv_cache1)
        #     self._detach_a_crossattn_cache(s.crossattn_cache)

        # self._detach_kv_cache()
        # self._detach_crossattn_cache()
        # for s in latent_states.values():
        #     if hasattr(s, "denoised_pred"):
        #         s.denoised_pred = s.denoised_pred.detach()
        #     s.initial_latent = s.initial_latent.detach()

        return sampled_critic_noise

    def tto_forward_backward(
        self,
        tto_trainer,
        latent_states: dict[str, SimpleNamespace],
        timestep,
        conditional_dict,
        index,
        block_index,
        batch_size,
        current_num_frames,
        noise,
        num_input_frames,
        unoptimized_spatial_similarities,
        tto_epoch,
        original_device,
        critic_denoised_pred: torch.Tensor | None = None,
        presampled_noise: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        # Create a temp copy of the latent state and move it to GPU. Only necessary items are copied.
        critic_latent_states: dict[str, SimpleNamespace] = {
            n: SimpleNamespace(
                noisy_input = s.noisy_input.detach().clone().to(original_device),
                current_start_frame = s.current_start_frame,
                initial_latent = s.initial_latent.detach().clone().to(original_device) if s.initial_latent is not None else None,
                crossattn_cache = self._clone_cache(s.crossattn_cache, original_device),
                kv_cache1 = self._clone_cache(s.kv_cache1, original_device)
            )
            for n, s in latent_states.items()}
        critic_timestep: torch.Tensor = timestep.detach().clone().to(original_device)

        sampled_critic_noise: torch.Tensor | None = None

        if tto_trainer.loss == "gaussian_forcing":
            tto_loss: torch.Tensor
            tto_loss, sampled_critic_noise = self.tto_compute_single_step_critic_gaussian_loss(
                tto_trainer,
                critic_latent_states,
                critic_timestep,
                conditional_dict,
                index,
                block_index,
                batch_size,
                current_num_frames,
                noise,
                num_input_frames,
                unoptimized_spatial_similarities,
                tto_epoch,
                critic_denoised_pred,
                presampled_noise=presampled_noise
            )
        else:
            tto_loss: torch.Tensor = self.tto_compute_loss(
                tto_trainer,
                critic_latent_states,
                critic_timestep,
                conditional_dict,
                index,
                block_index,
                batch_size,
                current_num_frames,
                noise,
                num_input_frames,
                unoptimized_spatial_similarities,
                tto_epoch
            )

        # torch.cuda.empty_cache()
        tto_trainer.backward_loss(tto_loss)

        self.vae.disable_gradient_checkpointing()

        if len(unoptimized_spatial_similarities) > 0:
            # Each denoising step is updated separately. No gradient should pass among timesteps either
            # from the KV caches or from the denoised prediction.
            unoptimized_spatial_similarities[block_index][index] = unoptimized_spatial_similarities[
                block_index][index].detach()

        return sampled_critic_noise

    def tto_compute_loss(
        self,
        tto_trainer,
        critic_latent_states,
        critic_timestep,
        conditional_dict,
        index,  # timestep index
        block_index,
        batch_size,
        current_num_frames,
        noise,
        num_input_frames,
        unoptimized_spatial_similarities,
        tto_epoch
    ) -> torch.Tensor:
        tto_trainer.optimizer.zero_grad(set_to_none=True)  # TODO: Call trainer.zero_grad
        # torch.cuda.empty_cache()  # TODO: Integrate it to trainer

        for n, s in critic_latent_states.items():
            _, s.denoised_pred = tto_trainer.model(
                noisy_image_or_video=s.noisy_input,
                conditional_dict=conditional_dict,
                timestep=critic_timestep,
                kv_cache=s.kv_cache1,
                crossattn_cache=s.crossattn_cache,
                current_start=s.current_start_frame * self.frame_seq_length
            )

        # In the case a separate critic model should be used instead of the updated model,
        # the outputs of it should be directly provided to the critic. The critic model is not
        # updated. The losses are computed on the output of the critic model.
        if tto_trainer.critic_model:
            # 1. Create a sneak peek of the next input to provide to the critic.
            if index < len(self.denoising_step_list) - 1:  # Same timeblock, next timestep.
                critic_next_timestep = self.denoising_step_list[index + 1]
                next_timestep_noise: Optional[torch.Tensor] = None
                for n, s in critic_latent_states.items():
                    if (tto_trainer.config.tto.get("perturbations_noise", None) != "same"
                            or next_timestep_noise is None):
                        # When the same noise corruption should be used for all the perturbed inputs,
                        # generate noise only once per timestep and use it on all inputs.
                        next_timestep_noise = torch.randn_like(s.denoised_pred.flatten(0, 1))

                    s.noisy_input = self.scheduler.add_noise(
                        s.denoised_pred.flatten(0, 1),
                        next_timestep_noise,
                        critic_next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, s.denoised_pred.shape[:2])
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        dist.broadcast(s.noisy_input, src=0)
            else:  # Next timeblock, first timestep.
                for n, s in critic_latent_states.items():
                    # Step 3.3: rerun with timestep zero to update KV cache using clean context
                    context_timestep = torch.ones_like(critic_timestep) * self.args.context_noise
                    tto_trainer.critic_model(
                        noisy_image_or_video=s.denoised_pred,
                        conditional_dict=conditional_dict,
                        timestep=context_timestep,
                        kv_cache=s.kv_cache1,
                        crossattn_cache=s.crossattn_cache,
                        current_start=s.current_start_frame * self.frame_seq_length,
                    )

                for n, s in critic_latent_states.items():
                    if s.current_start_frame < noise.size(dim=1):
                        # On the last timestep of the last block, there is no next input noise block. So,
                        # reuse the noise of the last block. If it is found to cause any issue, the extension
                        # of the input noise sequence by one block can be considered.
                        s.current_start_frame += current_num_frames
                critic_next_timestep = self.denoising_step_list[0]
                for n, s in critic_latent_states.items():
                    s.noisy_input = noise[
                        :, s.current_start_frame - 2*num_input_frames:s.current_start_frame + current_num_frames - 2*num_input_frames
                    ]

                # for n, s in critic_latent_states.items():
                #     _, s.denoised_pred = tto_trainer.critic_model(
                #         noisy_image_or_video=s.noisy_input,
                #         conditional_dict=conditional_dict,
                #         timestep=critic_timestep,
                #         kv_cache=s.cache1,
                #         crossattn_cache=s.crossattn_cache,
                #         current_start=s.current_start_frame * self.frame_seq_length
                #     )


            # 2. Execute the fake next step using the critic model.
            critic_timestep = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.int64) * critic_next_timestep

            for n, s in critic_latent_states.items():
                _, s.denoised_pred = tto_trainer.critic_model(
                    noisy_image_or_video=s.noisy_input,  # 1 step forward from the actual one
                    conditional_dict=conditional_dict,  # always remains the same
                    timestep=critic_timestep,  # 1 step forward from the actual one
                    kv_cache=s.kv_cache1,  # 1 step forward from the actual one
                    crossattn_cache=s.crossattn_cache,  # 1 step forward from the actual one
                    current_start=s.current_start_frame * self.frame_seq_length  # 1 step forward from the actual one
                )
        # ===== critic model processing end =====

        spatial_d_block: int = min(
            tto_trainer.config.tto.get("spatial_limit_blocks", block_index) + 1, block_index
        )
        print(f"Spatial_d block: {spatial_d_block} - Current block: {block_index}")
        tto_loss: torch.Tensor = losses.compute_tto_loss(
            tto_trainer.loss,
            {n: s.denoised_pred for n, s in critic_latent_states.items()},
            tto_trainer.masking_radius,
            unoptimized_spatial_similarities[spatial_d_block][index],
            critic_latent_states["original"].initial_latent,
            decoder=functools.partial(self.vae, mode="decode", use_cache=False),
            encoder=functools.partial(self.vae, mode="encode")
        )
        print(f"TTO loss (Epoch {tto_epoch}): {tto_loss.detach().cpu().item()}")

        return tto_loss

    def tto_compute_single_step_critic_gaussian_loss(
        self,
        tto_trainer: trainer.Trainer,
        critic_latent_states: dict[str, SimpleNamespace],
        critic_timestep,
        conditional_dict,
        index,  # timestep index
        block_index,
        batch_size,
        current_num_frames,
        noise,
        num_input_frames,
        unoptimized_spatial_similarities,
        tto_epoch: int,
        critic_denoised_pred: torch.Tensor | None = None,
        verbose: bool = True,
        presampled_noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tto_trainer.optimizer.zero_grad(set_to_none=True)  # TODO: Call trainer.zero_grad
        # torch.cuda.empty_cache()  # TODO: Integrate it to trainer

        batch_size: int = tto_trainer.config.tto.get("batch_size", batch_size)

        assert len(critic_latent_states) == 1  # Only original should be present.

        for n, s in critic_latent_states.items():
            _, s.denoised_pred = tto_trainer.model(
                noisy_image_or_video=s.noisy_input,
                conditional_dict=conditional_dict,
                timestep=critic_timestep,
                kv_cache=s.kv_cache1,
                crossattn_cache=s.crossattn_cache,
                current_start=s.current_start_frame * self.frame_seq_length
            )

        # ===== critic model processing start =====
        # In the case a separate critic model should be used instead of the updated model,
        # the outputs of it should be directly provided to the critic. The critic model is not
        # updated. The losses are computed on the output of the critic model.
        loss_reg: torch.Tensor | None = None
        if tto_trainer.critic_model:
            if critic_denoised_pred is not None and tto_trainer.config.tto.regularization == "critic_output_mse":
                loss_reg = torch.nn.functional.mse_loss(
                    critic_latent_states["original"].denoised_pred, critic_denoised_pred
                )
            elif critic_denoised_pred is not None:
                raise NotImplementedError(f"Not supported regularization: {tto_trainer.config.tto.regularization}")

            # Prepare latent states for the requested batch size.
            for s in critic_latent_states.values():
                s.crossattn_cache = self._crossatn_cache_batch_expand(s.crossattn_cache, batch_size)
                s.kv_cache1 = self._kv_cache_batch_repeat(s.kv_cache1, batch_size)
                _, t, c, h, w = s.denoised_pred.size()
                s.denoised_pred = s.denoised_pred.expand((batch_size, t, c, h, w))
                # s.initial_latent = s.initial_latent.expand((batch_size, t, c, h, w))
            conditional_dict = {"prompt_embeds": conditional_dict["prompt_embeds"].expand(batch_size, -1, -1)}

            # 1. Create a sneak peek of the next input to provide to the critic.
            if index < len(self.denoising_step_list) - 1:  # Same timeblock, next timestep.
                critic_next_timestep = self.denoising_step_list[index + 1]
                next_timestep_noise: Optional[torch.Tensor] = None
                for n, s in critic_latent_states.items():
                    if (tto_trainer.config.tto.get("perturbations_noise", None) != "same"
                            or next_timestep_noise is None):
                        # When the same noise corruption should be used for all the perturbed inputs,
                        # generate noise only once per timestep and use it on all inputs.
                        next_timestep_noise = (torch.randn_like(s.denoised_pred)
                                               if presampled_noise is None else presampled_noise)
                    sampled_critic_noise: torch.Tensor = next_timestep_noise.detach()
                    s.noisy_input = self.scheduler.add_noise(
                        s.denoised_pred.flatten(0, 1),
                        next_timestep_noise.flatten(0, 1),
                        critic_next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, s.denoised_pred.shape[:2])
                    # dist.broadcast(s.noisy_input, src=0)
            else:  # Next timeblock, first timestep.
                for n, s in critic_latent_states.items():
                    # Step 3.3: rerun with timestep zero to update KV cache using clean context
                    context_timestep = torch.ones_like(critic_timestep) * self.args.context_noise
                    context_timestep = context_timestep.expand(batch_size, -1)
                    tto_trainer.critic_model(
                        noisy_image_or_video=s.denoised_pred,
                        conditional_dict=conditional_dict,
                        timestep=context_timestep,
                        kv_cache=s.kv_cache1,
                        crossattn_cache=s.crossattn_cache,
                        current_start=s.current_start_frame * self.frame_seq_length,
                    )
                for n, s in critic_latent_states.items():
                    if s.current_start_frame < noise.size(dim=1):
                        # On the last timestep of the last block, there is no next input noise block. So,
                        # reuse the noise of the last block. If it is found to cause any issue, the extension
                        # of the input noise sequence by one block can be considered.
                        s.current_start_frame += current_num_frames
                critic_next_timestep = self.denoising_step_list[0]
                for n, s in critic_latent_states.items():
                    if presampled_noise is not None:
                        s.noisy_input = presampled_noise
                    elif tto_trainer.config.tto.get("critic_first_timestep_random_noise", False):
                        s.noisy_input = torch.randn_like(s.denoised_pred)
                    else:
                        s.noisy_input = noise[
                            :, s.current_start_frame - num_input_frames - current_num_frames:s.current_start_frame - num_input_frames
                        ]
                        _, t, c, h, w = s.noisy_input.size()
                        s.noisy_input = s.noisy_input.expand((batch_size, t, c, h, w))
                    sampled_critic_noise = s.noisy_input.detach()

                    # TODO: Implement such a mechanism that allows gradient checkpointing when rolling cache
                    #  directly to the KV cache implementation.
                    # ==== Fix for gradient checkpointing after rolling the cache. ====
                    # If cache is already full, increase its size to fit the tokens of another block and pass
                    # a view of it to the next critic step. This prevents overwriting past tokens and causing
                    # errors during the backward pass with gradient checkpointing that uses them, while
                    # allowing the model to always attend on the same number of tokens - if the size of the cache
                    # is naively increase, without taking a view, it won't.
                    kv_cache_size: int = s.kv_cache1[0]["k"].size(1)
                    kv_cache_end: int = s.kv_cache1[0]["local_end_index"][0].item()
                    latent_frames_per_block: int = 3  # TODO: Remove hardcoding.
                    cached_latent_frames: int = 21  # TODO: Remove hardcoding.
                    if kv_cache_size <= kv_cache_end:
                        s.current_start_frame -= latent_frames_per_block
                        for block_kv_cache in s.kv_cache1:
                            b, l, h, d = block_kv_cache["k"].size()
                            cache_device = block_kv_cache["k"].device
                            cache_dtype = block_kv_cache["k"].dtype
                            l = int((l / cached_latent_frames) * latent_frames_per_block)  # Expand by a block size.
                            k_extension: torch.Tensor = torch.zeros(
                                (b, l, h, d), device=cache_device, dtype=cache_dtype
                            )
                            block_kv_cache["k"] = torch.cat((block_kv_cache["k"], k_extension), dim=1)
                            block_kv_cache["k"] = block_kv_cache["k"][:, l:]
                            v_extension: torch.Tensor = torch.zeros(
                                (b, l, h, d), device=cache_device, dtype=cache_dtype
                            )
                            block_kv_cache["v"] = torch.cat((block_kv_cache["v"], v_extension), dim=1)
                            block_kv_cache["v"] = block_kv_cache["v"][:, l:]
                            block_kv_cache["global_end_index"] -= l
                            block_kv_cache["local_end_index"] -= l

            # 2. Execute the fake next step using the critic model.
            critic_timestep = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.int64) * critic_next_timestep

            for n, s in critic_latent_states.items():
                _, s.denoised_pred, s.pred_noise = tto_trainer.critic_model(
                    noisy_image_or_video=s.noisy_input,  # 1 step forward from the actual one
                    conditional_dict=conditional_dict,  # always remains the same
                    timestep=critic_timestep,  # 1 step forward from the actual one
                    kv_cache=s.kv_cache1,  # 1 step forward from the actual one
                    crossattn_cache=s.crossattn_cache,  # 1 step forward from the actual one
                    current_start=s.current_start_frame * self.frame_seq_length,  # 1 step forward from the actual one
                    return_pred_noise=True
                )
        # ===== critic model processing end =====

        loss: torch.Tensor
        per_moment_loss: dict[str, float]
        loss, per_moment_loss = tto_trainer.loss_fn(critic_latent_states["original"].pred_noise)

        gaussian_loss: torch.Tensor = loss
        reg_weight: float | None = None
        gaussian_weight: float | None = None
        if loss_reg is not None:
            reg_weight = tto_trainer.config.tto.regularization_weight
            gaussian_weight = tto_trainer.config.tto.white_gaussian_noise_weight
            loss = gaussian_weight * loss + reg_weight * loss_reg

        if verbose:
            print(f"Temporal block: {block_index} "
                  f"| Critic Flow Timestep: {critic_timestep[0, 0].detach().cpu().item()} "
                  f"| TTO Epoch: {tto_epoch}")
            print(f"\tTotal loss: {loss.detach().cpu().item()}")
            if loss_reg is not None:
                print(f"\tReg. Loss (w={reg_weight:.2f}): {reg_weight*loss_reg.detach().cpu().item()}")
                print(f"\tGauss. Loss (w={gaussian_weight:.2f}): {gaussian_weight*gaussian_loss.detach().cpu().item()}")
            print(f"\t\tMean loss: {per_moment_loss['loss_mean']}")
            print(f"\t\tVar loss: {per_moment_loss['loss_var']}")
            print(f"\t\tSkew loss: {per_moment_loss['loss_skew']}")
            print(f"\t\tKurt loss: {per_moment_loss['loss_kurt']}")
            print(f"\t\tSFM loss: {per_moment_loss['loss_sfm']}")
            if "loss_skew_low" in per_moment_loss:
                print(f"\t\tSkew loss (LP): {per_moment_loss['loss_skew_low']}")
            if "loss_kurt_low" in per_moment_loss:
                print(f"\t\tKurt loss (LP): {per_moment_loss['loss_kurt_low']}")

        return loss, sampled_critic_noise


    def tto_compute_single_step_critic_loss(
        self,
        tto_trainer,
        critic_latent_states,
        critic_timestep,
        conditional_dict,
        index,  # timestep index
        block_index,
        batch_size,
        current_num_frames,
        noise,
        num_input_frames,
        unoptimized_spatial_similarities,
        tto_epoch
    ) -> torch.Tensor:
        tto_trainer.optimizer.zero_grad(set_to_none=True)  # TODO: Call trainer.zero_grad
        # torch.cuda.empty_cache()  # TODO: Integrate it to trainer

        for n, s in critic_latent_states.items():
            _, s.denoised_pred = tto_trainer.model(
                noisy_image_or_video=s.noisy_input,
                conditional_dict=conditional_dict,
                timestep=critic_timestep,
                kv_cache=s.kv_cache1,
                crossattn_cache=s.crossattn_cache,
                current_start=s.current_start_frame * self.frame_seq_length
            )

        # TODO: Augment spectrally the s.denoised_pred.
        #  This requires creating new NamedInstance objects.
        # Perform low and high pass filtering of the denoised RGB frames.
        # masking_radius: int = tto_trainer.masking_radius
        # decoded_denoised_pred =
        # masked_initial_rgb = masking_strategies.spectral_mask_sequence(
        #     einops.rearrange(initial_rgb, "b c t h w -> b t c h w").float(),
        #     masking_radius=masking_radius
        # )
        # masked_initial_rgb = {k: einops.rearrange(
        #     v, "b t c h w -> b c t h w").to(dtype=initial_rgb.dtype)
        #                       for k, v in masked_initial_rgb.items()}
        # initial_rgb: dict[str, torch.Tensor] = {"original": initial_rgb}
        # if masked_initial_rgb is not None:
        #     initial_rgb.update(masked_initial_rgb)
        #     del masked_initial_rgb

        # In the case a separate critic model should be used instead of the updated model,
        # the outputs of it should be directly provided to the critic. The critic model is not
        # updated. The losses are computed on the output of the critic model.
        if tto_trainer.critic_model:
            # 1. Create a sneak peek of the next input to provide to the critic.
            if index < len(self.denoising_step_list) - 1:  # Same timeblock, next timestep.
                critic_next_timestep = self.denoising_step_list[index + 1]
                for n, s in critic_latent_states.items():
                    s.noisy_input = self.scheduler.add_noise(
                        s.denoised_pred.flatten(0, 1),
                        torch.randn_like(s.denoised_pred.flatten(0, 1)),
                        critic_next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, s.denoised_pred.shape[:2])
                    dist.broadcast(s.noisy_input, src=0)
            else:  # Next timeblock, first timestep.
                for n, s in critic_latent_states.items():
                    if s.current_start_frame < noise.size(dim=1):
                        # On the last timestep of the last block, there is no next input noise block. So,
                        # reuse the noise of the last block. If it is found to cause any issue, the extension
                        # of the input noise sequence by one block can be considered.
                        s.current_start_frame += current_num_frames
                critic_next_timestep = self.denoising_step_list[0]
                for n, s in critic_latent_states.items():
                    s.noisy_input = noise[
                        :, s.current_start_frame - 2*num_input_frames:s.current_start_frame + current_num_frames - 2*num_input_frames
                    ]

                for n, s in critic_latent_states.items():
                    # Step 3.3: rerun with timestep zero to update KV cache using clean context
                    context_timestep = torch.ones_like(critic_timestep) * self.args.context_noise
                    tto_trainer.critic_model(
                        noisy_image_or_video=s.denoised_pred,
                        conditional_dict=conditional_dict,
                        timestep=context_timestep,
                        kv_cache=s.kv_cache1,
                        crossattn_cache=s.crossattn_cache,
                        current_start=s.current_start_frame * self.frame_seq_length,
                    )

            # 2. Execute the fake next step using the critic model.
            critic_timestep = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.int64) * critic_next_timestep

            for n, s in critic_latent_states.items():
                _, s.denoised_pred = tto_trainer.critic_model(
                    noisy_image_or_video=s.noisy_input,  # 1 step forward from the actual one
                    conditional_dict=conditional_dict,  # always remains the same
                    timestep=critic_timestep,  # 1 step forward from the actual one
                    kv_cache=s.kv_cache1,  # 1 step forward from the actual one
                    crossattn_cache=s.crossattn_cache,  # 1 step forward from the actual one
                    current_start=s.current_start_frame * self.frame_seq_length  # 1 step forward from the actual one
                )
        # ===== critic model processing end =====

        spatial_d_block: int = min(
            tto_trainer.config.tto.get("spatial_limit_blocks", block_index) + 1, block_index
        )
        print(f"Spatial_d block: {spatial_d_block} - Current block: {block_index}")
        tto_loss: torch.Tensor = losses.compute_tto_loss(
            tto_trainer.loss,
            {n: s.denoised_pred for n, s in critic_latent_states.items()},
            tto_trainer.masking_radius,
            unoptimized_spatial_similarities[spatial_d_block][index],
            critic_latent_states["original"].initial_latent,
            decoder=functools.partial(self.vae, mode="decode", use_cache=False),
            encoder=functools.partial(self.vae, mode="encode")
        )
        print(f"TTO loss (Epoch {tto_epoch}): {tto_loss.detach().cpu().item()}")

        return tto_loss


    def _initialize_kv_cache(self, batch_size, dtype, device, num_caches: int = 1):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        self.kv_cache1 = []
        for _ in range(num_caches):
            kv_cache1 = []
            if self.local_attn_size != -1:
                # Use the local attention size to compute the KV cache size
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Use the default KV cache size
                kv_cache_size = 32760

            for _ in range(self.num_transformer_blocks):
                kv_cache1.append({
                    "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device, requires_grad=False),
                    "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device, requires_grad=False),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device, requires_grad=False),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device, requires_grad=False)
                })
            self.kv_cache1.append(kv_cache1) # always store the clean cache

        # Retain backward compatibility when num_caches == 1.
        if len(self.kv_cache1) == 1:
            self.kv_cache1 = self.kv_cache1[0]

    def _initialize_crossattn_cache(self, batch_size, dtype, device, num_caches: int = 1):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        self.crossattn_cache = []
        for _ in range(num_caches):
            crossattn_cache = []

            for _ in range(self.num_transformer_blocks):
                crossattn_cache.append({
                    "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device, requires_grad=False),
                    "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device, requires_grad=False),
                    "is_init": False
                })

            self.crossattn_cache.append(crossattn_cache)

        # Retain backward compatibility when num_caches == 1.
        if len(self.crossattn_cache) == 1:
            self.crossattn_cache = self.crossattn_cache[0]

    def _detach_kv_cache(self) -> None:
        for cache in self.kv_cache1:
            for i in range(self.num_transformer_blocks):
                cache[i]["k"].detach_()
                cache[i]["v"].detach_()
                cache[i]["global_end_index"].detach_()
                cache[i]["local_end_index"].detach_()

    def _detach_crossattn_cache(self) -> None:
        for cache in self.crossattn_cache:
            for i in range(self.num_transformer_blocks):
                cache[i]["k"].detach_()
                cache[i]["v"].detach_()

    def _detach_a_kv_cache(self, cache) -> None:
        for i in range(self.num_transformer_blocks):
            cache[i]["k"].detach_()
            cache[i]["v"].detach_()
            cache[i]["global_end_index"].detach_()
            cache[i]["local_end_index"].detach_()

    def _detach_a_crossattn_cache(self, cache) -> None:
        for i in range(self.num_transformer_blocks):
            cache[i]["k"].detach_()
            cache[i]["v"].detach_()

    def _move_kv_cache_to(self, cache: box.BoxList[box.Box], device: torch.device):
        for i in range(self.num_transformer_blocks):
            # 1. Get the item (this might be a temporary Box wrapper)
            block_cache = cache[i]

            # 2. Update the tensors (creates new tensor objects on the device)
            block_cache["k"] = block_cache["k"].to(device)
            block_cache["v"] = block_cache["v"].to(device)
            block_cache["global_end_index"] = block_cache["global_end_index"].to(device)
            block_cache["local_end_index"] = block_cache["local_end_index"].to(device)

            # 3. CRITICAL: Write the updated Box back to the list
            cache[i] = block_cache

    def _move_crossattn_cache_to(self, cache, device: torch.device):
        for i in range(self.num_transformer_blocks):
            block_cache = cache[i]

            block_cache["k"] = block_cache["k"].to(device)
            block_cache["v"] = block_cache["v"].to(device)

            cache[i] = block_cache

    def _clone_cache(
        self,
        cache: list[dict[str, torch.Tensor | bool]],
        device: torch.device
    ) -> list[dict[str, torch.Tensor]]:
        new_cache: list[dict[str, torch.Tensor]] = [
            {k: v.detach().clone().to(device).requires_grad_(False) if isinstance(v, torch.Tensor) else v for k, v in cache[i].items()}
            for i in range(self.num_transformer_blocks)
        ]
        return new_cache

    @staticmethod
    def _crossatn_cache_batch_expand(
        cache: list[dict[str, torch.Tensor | bool]],
        batch_size: int
    ) -> list[dict[str, torch.Tensor | bool]]:
        """Expands the batch dimension of a cross-attention cache in-place.

        :param cache: The cross-attention cache to be expanded.
        :param batch_size: The target batch size.

        :returns: The expanded cache.
        """
        for cached_block in cache:
            cached_block["k"] = cached_block["k"].expand((batch_size, -1, -1, -1))
            cached_block["v"] = cached_block["v"].expand((batch_size, -1, -1, -1))
        return cache

    @staticmethod
    def _kv_cache_batch_repeat(
        cache: list[dict[str, torch.Tensor]],
        batch_size: int
    ) -> list[dict[str, torch.Tensor]]:
        """Repeats the batch dimension of a kv-cache in-place.

        :param cache: The kv-cache to be repeated.
        :param batch_size: The target batch size.

        :returns: The repeated cache.
        """
        for cached_block in cache:
            cached_block["k"] = cached_block["k"].repeat((batch_size, 1, 1, 1))
            cached_block["v"] = cached_block["v"].repeat((batch_size, 1, 1, 1))
            cached_block["global_end_index"] = cached_block["global_end_index"].repeat(batch_size)
            cached_block["local_end_index"] = cached_block["local_end_index"].repeat(batch_size)
        return cache
