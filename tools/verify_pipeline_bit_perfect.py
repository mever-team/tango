"""Noise-floor-relative verification of `CausalInferenceTTOOptimizedPipeline`
against the reference `CausalInferenceTTOPipeline`.

GPU non-determinism (flash_attention, some cuDNN paths) prevents two
runs of even the *reference* pipeline from being bit-equal. This harness
therefore performs THREE runs of `generate(...)` on the same trainer
instance with identical inputs and identical starting state:

  Run A: reference pipeline (first invocation).
  Run B: reference pipeline (second invocation — establishes the GPU's
         noise floor at this configuration).
  Run C: optimized pipeline.

Between runs we restore exactly:
  * The LoRA params (a fresh kaiming/zero re-init, matching `reset_model`).
  * The optimizer state (cleared, since LoRA is reset).
  * The KV cache (re-allocated by `generate(...)` on each call).
  * The torch / numpy / cuda RNG state.
  * The noise tensor (sampled once and reused verbatim).
  * The conditioning latent (computed once and reused verbatim).

We then compute, per output (per_step_losses scalars, decoded video tensor,
latents tensor, LoRA params after the run), the maximum absolute deviation
of (A vs B) and (A vs C). The optimised pipeline is considered verified
when `max|A − C|  ≤  tolerance_factor * max|A − B|` for every comparison —
i.e. the optimisation introduces no more numerical drift than the GPU's
own run-to-run non-determinism, scaled by `tolerance_factor` (default 2).

`--reference_twice` runs A and B but skips C (a harness sanity check that
the reported "noise floor" makes sense by itself).

Usage (run on a host with the model checkpoint and one accessible GPU):

    python tools/verify_pipeline_bit_perfect.py \\
        --config_path configs/self_forcing_dmd_long_tto_gaussian_forcing_mse_reg_per_block_reset_lp_loss_lora.yaml \\
        --checkpoint_path checkpoints/self_forcing_dmd.pt \\
        --use_ema \\
        --video_path /home/dkarageo/datasets/<some_lvbench_video> \\
        --prompt "<the prompt>" \\
        --num_output_frames 15 \\
        --num_input_latent_frames 3 \\
        --tto_epochs 2

Defaults are tuned for a quick check that fits within ~3 generated blocks
(no KV-cache rolling). Use `--num_output_frames 24` or larger to cover the
rolling path (the optimized pipeline will fall back to cloning there and
should still match bit-perfectly).
"""

from __future__ import annotations

import argparse
import copy
import math
import pathlib
import sys

# Ensure the project root is on sys.path so the `pipeline` / `tto` packages
# import correctly regardless of how the script is invoked (e.g. `torchrun`
# from project root or direct python from anywhere).
_THIS_FILE = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.io import read_video

from pipeline.causal_inference_tto import CausalInferenceTTOPipeline
from pipeline.causal_inference_tto_optimized import CausalInferenceTTOOptimizedPipeline
from tto import lora as lora_mod
from tto import trainer as trainer_mod


# ---------------------------------------------------------------------------
# Comparison helpers.
# ---------------------------------------------------------------------------

def _fmt_tensor(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"


def _tensor_max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max absolute element-wise diff (float32 promotion to avoid bf16 overflow)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return float("inf")
    return float((a.cpu().float() - b.cpu().float()).abs().max().item())


def _losses_max_abs_diff(
    a: dict, b: dict,
) -> tuple[float, str]:
    """Max |a-b| over all leaf scalar entries in the nested per_step_losses
    structure. Returns (max_abs, location) for human-readable reporting."""
    if set(a.keys()) != set(b.keys()):
        return float("inf"), "top-level key mismatch"
    best = (0.0, "")
    for block in sorted(a.keys()):
        ab, bb = a[block], b[block]
        if set(ab.keys()) != set(bb.keys()):
            return float("inf"), f"step keys differ at block {block}"
        for step in sorted(ab.keys()):
            astep, bstep = ab[step], bb[step]
            if set(astep.keys()) != set(bstep.keys()):
                return float("inf"), f"epoch keys differ at block {block} step {step}"
            for epoch in sorted(astep.keys()):
                aep, bep = astep[epoch], bstep[epoch]
                if set(aep.keys()) != set(bep.keys()):
                    return float("inf"), f"component keys differ at block {block} step {step} ep {epoch}"
                for k in aep.keys():
                    d = abs(float(aep[k]) - float(bep[k]))
                    if d > best[0]:
                        best = (d, f"block={block} step={step} ep={epoch} {k}")
    return best


def _component_max_abs_diffs(
    a: dict, b: dict,
) -> dict[str, float]:
    """Per-loss-component max |a-b|, useful for spotting which loss term is
    the noisiest."""
    out: dict[str, float] = {}
    for block in a:
        for step in a[block]:
            for epoch in a[block][step]:
                for k, va in a[block][step][epoch].items():
                    vb = b[block][step][epoch][k]
                    d = abs(float(va) - float(vb))
                    if d > out.get(k, 0.0):
                        out[k] = d
    return out


# ---------------------------------------------------------------------------
# State snapshot / restore.
# ---------------------------------------------------------------------------

def snapshot_lora_state(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    """All `lora_A` / `lora_B` tensors, cloned to CPU."""
    return [
        (n, p.detach().clone().cpu())
        for n, p in model.named_parameters()
        if "lora_" in n
    ]


def restore_lora_state(model: torch.nn.Module, snap: list[tuple[str, torch.Tensor]]) -> None:
    by_name = dict(snap)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.copy_(by_name[n].to(p.device, dtype=p.dtype))


def snapshot_rng(device: torch.device) -> dict:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
        "cuda_all": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict, device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], device)
    torch.cuda.set_rng_state_all(state["cuda_all"])


# ---------------------------------------------------------------------------
# Pipeline construction.
# ---------------------------------------------------------------------------

def build_trainer_and_pipelines(args, config) -> tuple:
    """Build the trainer once, then construct *two* pipelines that share the
    same `generator`, `text_encoder`, `vae` instances. The reference
    pipeline will look at `tto_trainer.critic_model` (set up by the trainer)
    while the optimized pipeline ignores it and uses `lora.disabled(...)`."""
    tto_trainer = trainer_mod.Trainer(config)
    # Build the reference pipeline first — it instantiates the heavy modules.
    pipeline_ref = CausalInferenceTTOPipeline(config, device=tto_trainer.device)
    if args.checkpoint_path:
        sd = torch.load(args.checkpoint_path, map_location="cpu")
        pipeline_ref.generator.load_state_dict(sd["generator_ema" if args.use_ema else "generator"])
    pipeline_ref = pipeline_ref.to(dtype=tto_trainer.dtype)
    pipeline_ref.text_encoder.to(device=tto_trainer.device)
    pipeline_ref.vae.to(device=tto_trainer.device)
    tto_trainer.set_model(pipeline_ref.generator)

    # Build the optimized pipeline reusing the same modules. The trainer's
    # `set_model` already injected LoRA into `pipeline_ref.generator`; the
    # optimized pipeline shares that exact module (no re-injection).
    pipeline_opt = CausalInferenceTTOOptimizedPipeline(
        config,
        device=tto_trainer.device,
        generator=pipeline_ref.generator,
        text_encoder=pipeline_ref.text_encoder,
        vae=pipeline_ref.vae,
    )
    return tto_trainer, pipeline_ref, pipeline_opt


# ---------------------------------------------------------------------------
# Inputs.
# ---------------------------------------------------------------------------

def load_inputs(args, device, dtype, rgb_per_latent: int):
    """Load a single conditioning video + prompt + sample a noise tensor."""
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.Normalize([0.5], [0.5]),
    ])
    n_input_rgb = args.num_input_latent_frames * rgb_per_latent
    video, _, _ = read_video(args.video_path, output_format="TCHW", pts_unit="sec")
    end = args.visual_prompt_end_frame
    start = end - n_input_rgb
    video = video[start:end].float() / 255.0          # T C H W in [0,1]
    video = transform(video)                          # normalised [-1,1]
    video = video.unsqueeze(0).permute(0, 2, 1, 3, 4) # B C T H W
    video = video.to(device=device, dtype=dtype)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sampled_noise = torch.randn(
        [1, args.num_output_frames - args.num_input_latent_frames, 16, 60, 104],
        device=device, dtype=dtype,
    )
    return video, [args.prompt], sampled_noise


# ---------------------------------------------------------------------------
# Main verification driver.
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config_path", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--video_path", required=True, help="path to a single source mp4 to encode as conditioning")
    p.add_argument("--prompt", required=True)
    p.add_argument("--num_output_frames", type=int, default=15,
                   help="total latent frames including conditioning (15 = 4 generated blocks, no rolling)")
    p.add_argument("--num_input_latent_frames", type=int, default=3)
    p.add_argument("--visual_prompt_end_frame", type=int, default=93)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tto_epochs", type=int, default=2,
                   help="override tto.epochs to keep the check short")
    p.add_argument("--reference_twice", action="store_true",
                   help="run the REFERENCE pipeline twice only and report the noise floor "
                        "(skip the optimized run).")
    p.add_argument("--tolerance_factor", type=float, default=2.0,
                   help="optimised must not exceed this multiple of the reference-vs-reference "
                        "noise floor on any compared output. Default 2.0.")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="additional config overrides")
    args = p.parse_args()

    # Load + override config.
    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
    )
    if args.set:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.set))
    # Slim the inner loop so the run is quick. Bit-perfectness doesn't
    # depend on the count, just on the iteration being well-defined.
    OmegaConf.update(config, "tto.epochs", args.tto_epochs)

    # Force determinism. cuBLAS workspace config is set in the env outside.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # The trainer enables TF32 globally; override it before any model is
    # built so matmuls use the bit-stable fp32 path during verification.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("=== build trainer + pipelines ===", flush=True)
    tto_trainer, pipeline_ref, pipeline_opt = build_trainer_and_pipelines(args, config)
    # `Trainer.__init__` re-enables TF32; override AFTER it has run.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rgb_per_latent = config.get("rgb_frames_per_latent_frame", 3)
    video_in, prompts, noise = load_inputs(args, tto_trainer.device, tto_trainer.dtype, rgb_per_latent)

    print("=== snapshot LoRA initial state ===", flush=True)
    initial_lora_snap = snapshot_lora_state(pipeline_ref.generator)
    tto_trainer.optimizer.state.clear()
    pre_rng = snapshot_rng(tto_trainer.device)

    def run(pipeline, label: str):
        print(f"\n=== Run {label} ===", flush=True)
        # Restore state to the initial snapshot so all runs see identical
        # starting conditions.
        restore_lora_state(pipeline_ref.generator, initial_lora_snap)
        tto_trainer.optimizer.state.clear()
        restore_rng(pre_rng, tto_trainer.device)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        decoded, latents, stats = pipeline.generate(
            tto_trainer=tto_trainer,
            noise=noise.clone(),
            text_prompts=list(prompts),
            initial_rgb=video_in.clone(),
            return_latents=True,
        )
        lora_after = snapshot_lora_state(pipeline_ref.generator)
        return {
            "decoded": decoded,
            "latents": latents,
            "per_step_losses": stats["per_step_losses"],
            "lora_after": lora_after,
        }

    # -----------------------------------------------------------------
    # 3 runs: reference x 2 (noise-floor baseline) + optimized.
    # -----------------------------------------------------------------
    A = run(pipeline_ref, "A: reference (first)")
    B = run(pipeline_ref, "B: reference (second)")
    if not args.reference_twice:
        C = run(pipeline_opt, "C: optimized")
    else:
        C = None

    # -----------------------------------------------------------------
    # Compute max-abs deviations.
    # -----------------------------------------------------------------
    print("\n=== noise floor (A vs B reference runs) and signal (A vs C optimized) ===", flush=True)

    def lora_max_diff(a_lora, b_lora) -> tuple[float, str]:
        if len(a_lora) != len(b_lora):
            return float("inf"), "param count mismatch"
        best = (0.0, "")
        for (na, pa), (nb, pb) in zip(a_lora, b_lora):
            if na != nb:
                return float("inf"), f"name mismatch {na} vs {nb}"
            d = _tensor_max_abs_diff(pa, pb)
            if d > best[0]:
                best = (d, na)
        return best

    floor_losses, floor_loc = _losses_max_abs_diff(A["per_step_losses"], B["per_step_losses"])
    floor_decoded = _tensor_max_abs_diff(A["decoded"], B["decoded"])
    floor_latents = _tensor_max_abs_diff(A["latents"], B["latents"]) if A["latents"] is not None else 0.0
    floor_lora, floor_lora_loc = lora_max_diff(A["lora_after"], B["lora_after"])

    print(f"  per_step_losses   noise floor max |Δ| = {floor_losses:.3e}  (at {floor_loc})")
    print(f"  decoded_video     noise floor max |Δ| = {floor_decoded:.3e}")
    print(f"  latents           noise floor max |Δ| = {floor_latents:.3e}")
    print(f"  lora_params       noise floor max |Δ| = {floor_lora:.3e}  (at {floor_lora_loc})")

    per_comp_floor = _component_max_abs_diffs(A["per_step_losses"], B["per_step_losses"])
    print("  per-component noise floor (sorted):")
    for k in sorted(per_comp_floor, key=lambda kk: -per_comp_floor[kk]):
        print(f"    {k:<20} {per_comp_floor[k]:.3e}")

    if C is None:
        print("\n(--reference_twice set; skipping optimised run comparison)")
        return 0

    sig_losses, sig_loc = _losses_max_abs_diff(A["per_step_losses"], C["per_step_losses"])
    sig_decoded = _tensor_max_abs_diff(A["decoded"], C["decoded"])
    sig_latents = _tensor_max_abs_diff(A["latents"], C["latents"]) if A["latents"] is not None else 0.0
    sig_lora, sig_lora_loc = lora_max_diff(A["lora_after"], C["lora_after"])

    print(f"\n  per_step_losses   optimised max |Δ| vs A = {sig_losses:.3e}  (at {sig_loc})")
    print(f"  decoded_video     optimised max |Δ| vs A = {sig_decoded:.3e}")
    print(f"  latents           optimised max |Δ| vs A = {sig_latents:.3e}")
    print(f"  lora_params       optimised max |Δ| vs A = {sig_lora:.3e}  (at {sig_lora_loc})")

    per_comp_sig = _component_max_abs_diffs(A["per_step_losses"], C["per_step_losses"])
    print("  per-component optimised drift (sorted):")
    for k in sorted(per_comp_sig, key=lambda kk: -per_comp_sig[kk]):
        floor_k = per_comp_floor.get(k, 0.0)
        ratio = (per_comp_sig[k] / floor_k) if floor_k > 0 else float("inf") if per_comp_sig[k] > 0 else 0.0
        marker = "" if ratio <= args.tolerance_factor else "  ← exceeds tolerance"
        print(f"    {k:<20} {per_comp_sig[k]:.3e}  (noise floor: {floor_k:.3e}, ratio: {ratio:.2f}x){marker}")

    # -----------------------------------------------------------------
    # PASS / FAIL.
    # -----------------------------------------------------------------
    tf = args.tolerance_factor
    failures: list[str] = []
    def check(name, sig, floor):
        # If the noise floor is zero (the output is bit-stable), we require
        # the optimised run to also be bit-stable. Otherwise the optimised
        # drift must be ≤ tf × floor.
        if floor == 0:
            if sig != 0:
                failures.append(f"{name}: optimised drift {sig:.3e} but noise floor is 0")
        else:
            if sig > tf * floor:
                failures.append(f"{name}: optimised {sig:.3e} > {tf}x floor ({floor:.3e}); ratio {sig/floor:.2f}x")
    check("per_step_losses", sig_losses, floor_losses)
    check("decoded_video",   sig_decoded, floor_decoded)
    check("latents",         sig_latents, floor_latents)
    check("lora_params",     sig_lora,    floor_lora)

    if failures:
        print(f"\n!!! VERIFICATION FAILED ({len(failures)} channel(s) exceed tolerance):")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print(f"\n+++ VERIFIED +++ all optimised drifts are within {tf:.1f}× the GPU noise floor.")
    print(f"  per_step_losses     {sig_losses:.3e}  ≤  {tf}× {floor_losses:.3e}  ({floor_losses*tf:.3e})")
    print(f"  decoded_video       {sig_decoded:.3e}  ≤  {tf}× {floor_decoded:.3e}")
    print(f"  latents             {sig_latents:.3e}  ≤  {tf}× {floor_latents:.3e}")
    print(f"  lora_params         {sig_lora:.3e}  ≤  {tf}× {floor_lora:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
