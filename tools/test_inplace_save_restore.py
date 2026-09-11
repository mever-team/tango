"""Unit-test the rolling-case `_save_kv_cache_for_inplace` /
`_restore_kv_cache_for_inplace` pair end-to-end against a hand-coded
simulation of `wan/modules/causal_model.py:206-235`.

This exercises ONLY the cache-state arithmetic — it does not run the
model, so it is fast (~seconds) and independent of LoRA / checkpoints
/ VAEs etc. If this test passes, the chunked right-to-left in-place
copy correctly inverts the model's shift-then-write for any
(num_evicted, num_rolled) split.

Usage:
    python tools/test_inplace_save_restore.py
"""

from __future__ import annotations

import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from pipeline.causal_inference_tto_optimized import CausalInferenceTTOOptimizedPipeline


def _make_fake_cache(cache_size: int, num_layers: int, device, dtype):
    """A list of layer dicts matching the schema the pipeline expects."""
    torch.manual_seed(0)
    return [
        {
            "k": torch.randn(1, cache_size, 12, 128, device=device, dtype=dtype),
            "v": torch.randn(1, cache_size, 12, 128, device=device, dtype=dtype),
            "local_end_index": torch.tensor([cache_size], dtype=torch.long, device=device),
            "global_end_index": torch.tensor([cache_size], dtype=torch.long, device=device),
        }
        for _ in range(num_layers)
    ]


def _simulate_model_roll(
    cache: list[dict],
    sink: int,
    num_new: int,
    new_k_v_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    current_end: int,
) -> None:
    """Apply the EXACT operation from `wan/modules/causal_model.py:206-235`
    that rolls the cache: shift-then-write."""
    for entry, (new_k, new_v) in zip(cache, new_k_v_pairs):
        cache_size = entry["k"].size(1)
        local_end = int(entry["local_end_index"][0].item())
        num_evicted = num_new + local_end - cache_size
        num_rolled = local_end - num_evicted - sink
        # The model does an explicit .clone() to avoid aliased writes.
        entry["k"][:, sink:sink + num_rolled] = (
            entry["k"][:, sink + num_evicted:sink + num_evicted + num_rolled].clone()
        )
        entry["v"][:, sink:sink + num_rolled] = (
            entry["v"][:, sink + num_evicted:sink + num_evicted + num_rolled].clone()
        )
        local_end_index = (
            local_end + current_end - int(entry["global_end_index"][0].item()) - num_evicted
        )
        local_start_index = local_end_index - num_new
        entry["k"][:, local_start_index:local_end_index] = new_k
        entry["v"][:, local_start_index:local_end_index] = new_v
        entry["local_end_index"].fill_(local_end_index)
        entry["global_end_index"].fill_(current_end)


class _FakePipeline:
    """Just enough surface for `_save_kv_cache_for_inplace` to run."""
    FRAME_SEQ_LENGTH = 1560

    def __init__(self, local_attn_size: int, sink_size: int = 0):
        self.local_attn_size = local_attn_size

        class _Model:
            pass

        self.generator = _Model()
        self.generator.model = _Model()
        self.generator.model.sink_size = sink_size

    save = CausalInferenceTTOOptimizedPipeline._save_kv_cache_for_inplace
    restore = staticmethod(CausalInferenceTTOOptimizedPipeline._restore_kv_cache_for_inplace)
    _sink_tokens = CausalInferenceTTOOptimizedPipeline._sink_tokens


def _diff(a, b) -> float:
    return float((a.cpu().float() - b.cpu().float()).abs().max().item())


def test_canonical_rolling():
    """Canonical config: cache_size=32760, num_new=4680, num_evicted=4680,
    num_rolled=28080, sink=0. Six chunks of size 4680 each."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    cache_size = 32760
    num_layers = 4  # small for speed
    num_new = 4680
    sink = 0

    cache = _make_fake_cache(cache_size, num_layers, device, dtype)
    originals = [
        {"k": e["k"].clone(), "v": e["v"].clone(),
         "local_end_index": e["local_end_index"].clone(),
         "global_end_index": e["global_end_index"].clone()}
        for e in cache
    ]

    pipe = _FakePipeline(local_attn_size=21, sink_size=0)

    saved = pipe.save(
        cache,
        current_start_frame=cache_size // _FakePipeline.FRAME_SEQ_LENGTH,
        num_new_frames=num_new // _FakePipeline.FRAME_SEQ_LENGTH,
    )
    # Every layer must take the "rolling" branch.
    for snap in saved["layers"]:
        assert snap["strategy"] == "rolling", f"expected rolling, got {snap['strategy']}"

    # Simulate model's shift-then-write.
    torch.manual_seed(42)
    new_kvs = [
        (torch.randn(1, num_new, 12, 128, device=device, dtype=dtype),
         torch.randn(1, num_new, 12, 128, device=device, dtype=dtype))
        for _ in range(num_layers)
    ]
    current_end = cache_size + num_new  # global_end was cache_size; advance by num_new
    _simulate_model_roll(cache, sink=sink, num_new=num_new,
                         new_k_v_pairs=new_kvs, current_end=current_end)

    # Sanity: state should be DIFFERENT from original now.
    pre_restore_diff = max(_diff(cache[i]["k"], originals[i]["k"]) for i in range(num_layers))
    assert pre_restore_diff > 0, "model didn't actually mutate the cache"

    # Restore.
    pipe.restore(cache, saved)

    # Verify byte-equal to original (bf16 has 0 noise on a pure memcpy).
    for i, (e, o) in enumerate(zip(cache, originals)):
        k_diff = _diff(e["k"], o["k"])
        v_diff = _diff(e["v"], o["v"])
        le_diff = int((e["local_end_index"] - o["local_end_index"]).abs().max().item())
        ge_diff = int((e["global_end_index"] - o["global_end_index"]).abs().max().item())
        assert k_diff == 0, f"layer {i}: k_diff = {k_diff} (expected 0)"
        assert v_diff == 0, f"layer {i}: v_diff = {v_diff} (expected 0)"
        assert le_diff == 0, f"layer {i}: local_end diff = {le_diff}"
        assert ge_diff == 0, f"layer {i}: global_end diff = {ge_diff}"

    print(f"test_canonical_rolling: OK  (pre-restore diff was {pre_restore_diff:.3e})")


def test_partial_chunk_rolling():
    """num_rolled NOT divisible by num_evicted — exercises the last
    partial chunk in the chunked copy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Configure so num_evicted=1000, num_rolled=2500 (not a multiple).
    cache_size = 3500
    num_layers = 2
    num_new = 1500  # so num_evicted = 1500 + 3500 - 3500 = 1500. Hmm let me redo.
    # Want num_evicted=1000, num_rolled=2500. local_end=cache_size=3500.
    # num_evicted = num_new + local_end - cache_size = num_new.
    # So num_new = 1000 → num_evicted=1000. num_rolled = local_end - num_evicted - sink = 3500 - 1000 = 2500.
    num_new = 1000

    cache = _make_fake_cache(cache_size, num_layers, device, dtype)
    originals = [
        {"k": e["k"].clone(), "v": e["v"].clone(),
         "local_end_index": e["local_end_index"].clone(),
         "global_end_index": e["global_end_index"].clone()}
        for e in cache
    ]

    pipe = _FakePipeline(local_attn_size=21, sink_size=0)

    # Hijack FRAME_SEQ_LENGTH to 1 so we can pass num_new directly as num_new_frames.
    pipe.FRAME_SEQ_LENGTH = 1
    saved = CausalInferenceTTOOptimizedPipeline._save_kv_cache_for_inplace(
        pipe, cache, current_start_frame=cache_size, num_new_frames=num_new,
    )
    for snap in saved["layers"]:
        assert snap["strategy"] == "rolling", f"expected rolling, got {snap['strategy']}"
        assert snap["num_evicted"] == 1000
        assert snap["num_rolled"] == 2500

    torch.manual_seed(43)
    new_kvs = [
        (torch.randn(1, num_new, 12, 128, device=device, dtype=dtype),
         torch.randn(1, num_new, 12, 128, device=device, dtype=dtype))
        for _ in range(num_layers)
    ]
    current_end = cache_size + num_new
    _simulate_model_roll(cache, sink=0, num_new=num_new,
                         new_k_v_pairs=new_kvs, current_end=current_end)

    CausalInferenceTTOOptimizedPipeline._restore_kv_cache_for_inplace(cache, saved)

    for i, (e, o) in enumerate(zip(cache, originals)):
        k_diff = _diff(e["k"], o["k"])
        v_diff = _diff(e["v"], o["v"])
        assert k_diff == 0, f"layer {i}: k_diff = {k_diff}"
        assert v_diff == 0, f"layer {i}: v_diff = {v_diff}"

    print("test_partial_chunk_rolling: OK  (num_evicted=1000, num_rolled=2500, 3 chunks of [1000,1000,500])")


def test_no_roll_branch():
    """current_end == global_end (sub-timestep > 0 case): model writes
    at [local_end - num_new : local_end] without shifting. Save/restore
    should preserve the cache."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    cache_size = 32760
    num_layers = 4
    num_new = 4680

    cache = _make_fake_cache(cache_size, num_layers, device, dtype)
    # global_end_index initially = cache_size, local_end_index = cache_size.
    # Set up: simulate "after sub-timestep 0's actual_denoise_fwd advanced
    # global_end to cache_size + num_new but the LIVE cache state remains
    # full at cache_size local tokens".
    # In reality: model already shifted+wrote at sub-timestep 0; current
    # state has local_end=cache_size, global_end=cache_size+num_new.
    for e in cache:
        e["global_end_index"].fill_(cache_size + num_new)
    originals = [
        {"k": e["k"].clone(), "v": e["v"].clone(),
         "local_end_index": e["local_end_index"].clone(),
         "global_end_index": e["global_end_index"].clone()}
        for e in cache
    ]

    pipe = _FakePipeline(local_attn_size=21, sink_size=0)

    # current_start = cache_size (in tokens), num_new_frames * FRAME_SEQ_LENGTH = num_new tokens.
    # current_end = cache_size + num_new == global_end. Should be no_roll.
    saved = CausalInferenceTTOOptimizedPipeline._save_kv_cache_for_inplace(
        pipe, cache,
        current_start_frame=cache_size // _FakePipeline.FRAME_SEQ_LENGTH,
        num_new_frames=num_new // _FakePipeline.FRAME_SEQ_LENGTH,
    )
    for snap in saved["layers"]:
        assert snap["strategy"] == "no_roll", f"expected no_roll, got {snap['strategy']}"

    # Simulate model's no-roll branch.
    torch.manual_seed(44)
    for entry in cache:
        local_end = int(entry["local_end_index"][0].item())
        local_end_post = local_end  # current_end == global_end → local_end_index unchanged
        local_start_post = local_end_post - num_new
        new_k = torch.randn(1, num_new, 12, 128, device=device, dtype=dtype)
        new_v = torch.randn(1, num_new, 12, 128, device=device, dtype=dtype)
        entry["k"][:, local_start_post:local_end_post] = new_k
        entry["v"][:, local_start_post:local_end_post] = new_v
        # No index change for current_end == global_end.

    CausalInferenceTTOOptimizedPipeline._restore_kv_cache_for_inplace(cache, saved)

    for i, (e, o) in enumerate(zip(cache, originals)):
        k_diff = _diff(e["k"], o["k"])
        v_diff = _diff(e["v"], o["v"])
        assert k_diff == 0, f"layer {i}: k_diff = {k_diff}"
        assert v_diff == 0, f"layer {i}: v_diff = {v_diff}"

    print("test_no_roll_branch: OK")


if __name__ == "__main__":
    test_canonical_rolling()
    test_partial_chunk_rolling()
    test_no_roll_branch()
    print("\nALL TESTS PASSED.")
