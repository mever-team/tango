"""Regression tests for the warped denoising schedule.

Background
----------
`denoising_step_list` starts as integers [1000, 750, 500, 250] but, when
`warp_denoising_step` is set, is re-indexed through `FlowMatchScheduler.timesteps`
and becomes **non-integer**: [1000.0, 937.5, 833.333, 625.0].

Those values are not just labels. `WanDiffusionWrapper._convert_flow_pred_to_x0`
and `FlowMatchScheduler.add_noise` both recover sigma by an `argmin` over the
1000-entry timestep grid, and 937.5 / 833.333 sit *exactly* on grid points.
Truncating them to int (937 / 833) makes the argmin resolve to a neighbouring
grid entry, so the pipeline denoises at a slightly different noise level than the
schedule specifies — and differs from `CausalInferencePipeline`, which preserves
the float via type promotion.

Both TTO pipelines used to call `int(...)` on these values and build the timestep
tensors with an integer dtype. These tests pin the corrected behaviour:

  1. `TestWarpedScheduleMath`      — the schedule arithmetic, and what truncation costs.
  2. `TestPipelinesDoNotTruncate`  — source-level guard against the regression.
  3. `TestSchedulerRoundTrip`      — end-to-end against the real torch scheduler.

Classes 1 and 2 need only the standard library, so they run anywhere. Class 3 is
skipped when torch is unavailable.

Run:  python -m unittest discover -s tests -v
"""

import ast
import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TTO_PIPELINES = (
    "pipeline/causal_inference_tto.py",
    "pipeline/causal_inference_tto_optimized.py",
)

# Schedule parameters as configured for Self-Forcing (configs/*.yaml:
# model_kwargs.timestep_shift = 5.0) and applied in utils/wan_wrapper.py.
SHIFT = 5.0
NUM_TRAIN_TIMESTEPS = 1000
DENOISING_STEP_LIST = [1000, 750, 500, 250]

# The warped timesteps the pipelines must use, and their sigmas.
EXPECTED_TIMESTEPS = [1000.0, 937.5, 833.3333333333334, 625.0]
EXPECTED_SIGMAS = [1.0, 0.9375, 0.8333333333333334, 0.625]


def build_sigma_grid(shift=SHIFT, n=NUM_TRAIN_TIMESTEPS):
    """Reimplementation of FlowMatchScheduler.set_timesteps (utils/scheduler.py).

    sigma_min=0.0, sigma_max=1.0, extra_one_step=True, denoising_strength=1.0:
        sigmas = linspace(1.0, 0.0, n + 1)[:-1]        -> n entries
        sigmas = shift * s / (1 + (shift - 1) * s)     -> the warp
        timesteps = sigmas * n
    """
    raw = [1.0 - i / n for i in range(n)]
    sigmas = [shift * s / (1 + (shift - 1) * s) for s in raw]
    timesteps = [s * n for s in sigmas]
    return sigmas, timesteps


def warp_denoising_steps(step_list=DENOISING_STEP_LIST):
    """Reimplementation of the `warp_denoising_step` branch in the pipelines:
    `timesteps[1000 - denoising_step_list]`, with a 0 appended to `timesteps`.
    """
    _, timesteps = build_sigma_grid()
    table = timesteps + [0.0]
    return [table[NUM_TRAIN_TIMESTEPS - s] for s in step_list]


def sigma_for_timestep(t):
    """Reimplementation of the argmin lookup in
    `WanDiffusionWrapper._convert_flow_pred_to_x0` / `FlowMatchScheduler.add_noise`.
    """
    sigmas, timesteps = build_sigma_grid()
    idx = min(range(len(timesteps)), key=lambda i: abs(timesteps[i] - t))
    return sigmas[idx]


class TestWarpedScheduleMath(unittest.TestCase):
    """The schedule arithmetic, and the cost of truncating it."""

    def test_warped_timesteps_are_non_integer(self):
        got = warp_denoising_steps()
        for expected, actual in zip(EXPECTED_TIMESTEPS, got):
            self.assertAlmostEqual(expected, actual, places=6)
        # The whole point: two of the four are NOT integers.
        self.assertNotEqual(got[1], int(got[1]), "937.5 must stay non-integer")
        self.assertNotEqual(got[2], int(got[2]), "833.33 must stay non-integer")

    def test_exact_timesteps_resolve_to_exact_sigmas(self):
        """The untruncated timesteps sit exactly on grid points."""
        for t, expected_sigma in zip(warp_denoising_steps(), EXPECTED_SIGMAS):
            self.assertAlmostEqual(sigma_for_timestep(t), expected_sigma, places=9)

    def test_sigma_zero_is_bit_exact_one(self):
        """sigma at the first denoising step is exactly 1.0.

        This is load-bearing elsewhere: pred_noise = (1 - sigma) * flow + x_t, so
        sigma == 1.0 makes pred_noise independent of the model.
        """
        self.assertEqual(sigma_for_timestep(warp_denoising_steps()[0]), 1.0)

    def test_truncation_shifts_sigma(self):
        """Documents the regression this suite guards against."""
        exact = warp_denoising_steps()
        truncated = [float(int(t)) for t in exact]

        # Steps 0 and 3 are already integers, so truncation is a no-op there.
        self.assertEqual(sigma_for_timestep(exact[0]), sigma_for_timestep(truncated[0]))
        self.assertEqual(sigma_for_timestep(exact[3]), sigma_for_timestep(truncated[3]))

        # Steps 1 and 2 land on a different grid entry.
        for i in (1, 2):
            self.assertNotAlmostEqual(
                sigma_for_timestep(exact[i]), sigma_for_timestep(truncated[i]), places=6,
                msg=f"truncating step {i} must change the resolved sigma",
            )

    def test_truncation_error_in_the_lookahead_coefficient(self):
        """The model-dependent coefficient is c = (1 - sigma) / sigma.

        Truncation inflates it by ~1.07% and ~0.40% at the two affected steps.
        """
        def coeff(t):
            s = sigma_for_timestep(t)
            return (1.0 - s) / s

        exact = warp_denoising_steps()
        for i, expected_pct in ((1, 1.07), (2, 0.40)):
            c_exact = coeff(exact[i])
            c_trunc = coeff(float(int(exact[i])))
            pct = (c_trunc - c_exact) / c_exact * 100.0
            self.assertAlmostEqual(pct, expected_pct, delta=0.05)


class TestPipelinesDoNotTruncate(unittest.TestCase):
    """Source-level guard: no TTO pipeline may truncate a warped timestep.

    Catches both historical spellings:
        int(self.denoising_step_list[i])
        for idx, t_val in enumerate(self.denoising_step_list): int(t_val)
    and any timestep tensor built with an integer dtype.
    """

    @staticmethod
    def _timestep_names(tree):
        """Names bound to an element of `self.denoising_step_list`."""
        names = set()
        for node in ast.walk(tree):
            # for idx, t_val in enumerate(self.denoising_step_list)
            if isinstance(node, ast.For):
                it = ast.dump(node.iter)
                if "denoising_step_list" in it:
                    tgt = node.target
                    elts = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
                    for e in elts:
                        if isinstance(e, ast.Name):
                            names.add(e.id)
            # x = self.denoising_step_list[i]
            if isinstance(node, ast.Assign) and "denoising_step_list" in ast.dump(node.value):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
            if isinstance(node, ast.AnnAssign) and node.value is not None:
                if "denoising_step_list" in ast.dump(node.value) and isinstance(node.target, ast.Name):
                    names.add(node.target.id)
        # `idx` from enumerate is an index, not a timestep
        return {n for n in names if n not in {"idx", "index", "i"}}

    def test_no_int_truncation_of_warped_timesteps(self):
        for rel in TTO_PIPELINES:
            path = os.path.join(REPO_ROOT, rel)
            with open(path) as fh:
                src = fh.read()
            tree = ast.parse(src)
            ts_names = self._timestep_names(tree)
            offenders = []
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                    continue
                if node.func.id not in {"int", "round"} or not node.args:
                    continue
                arg = node.args[0]
                seg = ast.get_source_segment(src, arg) or ast.dump(arg)
                hits_list = "denoising_step_list" in seg
                hits_name = isinstance(arg, ast.Name) and arg.id in ts_names
                if hits_list or hits_name:
                    offenders.append(f"{rel}:{node.lineno}: {node.func.id}({seg})")
            self.assertEqual(
                offenders, [],
                "Warped timesteps must not be truncated — they are non-integer "
                "(937.5, 833.33) and int() shifts the sigma the scheduler resolves.\n"
                + "\n".join(offenders),
            )

    def test_timestep_tensors_are_not_integer_dtype(self):
        """A tensor whose fill value is a warped timestep must not be int64/long.

        `torch.full(..., 937.5, dtype=torch.int64)` would silently re-truncate.
        """
        for rel in TTO_PIPELINES:
            path = os.path.join(REPO_ROOT, rel)
            with open(path) as fh:
                src = fh.read()
            tree = ast.parse(src)
            ts_names = self._timestep_names(tree)
            offenders = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = ast.get_source_segment(src, node.func) or ""
                if not fn.endswith(("torch.full", "torch.full_like", "torch.ones", "torch.zeros")):
                    continue
                args_src = " ".join(
                    (ast.get_source_segment(src, a) or "") for a in node.args
                )
                mentions_ts = "denoising_step_list" in args_src or any(
                    n in args_src.split() or n + "," in args_src for n in ts_names
                )
                if not mentions_ts:
                    continue
                for kw in node.keywords:
                    if kw.arg != "dtype":
                        continue
                    dt = ast.get_source_segment(src, kw.value) or ""
                    if dt.endswith(("int64", "long", "int32", "int")):
                        offenders.append(f"{rel}:{node.lineno}: dtype={dt}")
            self.assertEqual(
                offenders, [],
                "Timestep tensors must be float — an integer dtype re-truncates "
                "the warped value.\n" + "\n".join(offenders),
            )

    def test_base_pipeline_still_preserves_float_timesteps(self):
        """The base pipeline is the reference behaviour; it must not regress either."""
        path = os.path.join(REPO_ROOT, "pipeline/causal_inference.py")
        with open(path) as fh:
            src = fh.read()
        tree = ast.parse(src)
        offenders = [
            f"causal_inference.py:{n.lineno}"
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "int"
            and n.args
            and "denoising_step_list" in (ast.get_source_segment(src, n.args[0]) or "")
        ]
        self.assertEqual(offenders, [], "\n".join(offenders))


class TestSchedulerRoundTrip(unittest.TestCase):
    """End-to-end checks against the real torch scheduler, when torch is available."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            from utils.scheduler import FlowMatchScheduler  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"torch/scheduler unavailable: {exc}")

    def _scheduler(self):
        from utils.scheduler import FlowMatchScheduler
        # Mirrors utils/wan_wrapper.py's construction.
        sch = FlowMatchScheduler(
            shift=SHIFT, sigma_min=0.0, extra_one_step=True,
        )
        sch.set_timesteps(NUM_TRAIN_TIMESTEPS, training=True)
        return sch

    def test_warped_timesteps_match_the_pure_python_model(self):
        import torch
        sch = self._scheduler()
        table = torch.cat((sch.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        idx = torch.tensor([NUM_TRAIN_TIMESTEPS - s for s in DENOISING_STEP_LIST])
        warped = table[idx].tolist()
        for expected, actual in zip(EXPECTED_TIMESTEPS, warped):
            self.assertAlmostEqual(expected, actual, places=3)

    def test_add_noise_uses_the_exact_sigma(self):
        """add_noise(x0, noise, t) == (1 - sigma) * x0 + sigma * noise."""
        import torch
        sch = self._scheduler()
        x0 = torch.zeros(1, 4, 4, 4)
        noise = torch.ones(1, 4, 4, 4)
        for t, sigma in zip(EXPECTED_TIMESTEPS, EXPECTED_SIGMAS):
            out = sch.add_noise(x0, noise, torch.tensor([t], dtype=torch.float32))
            # x0 = 0, noise = 1  =>  every element equals sigma
            self.assertAlmostEqual(out.flatten()[0].item(), sigma, places=4)

    def test_truncated_timestep_gives_a_different_sigma(self):
        """The regression, demonstrated through the real scheduler."""
        import torch
        sch = self._scheduler()
        x0 = torch.zeros(1, 2, 2, 2)
        noise = torch.ones(1, 2, 2, 2)

        def sigma_of(t):
            return sch.add_noise(
                x0, noise, torch.tensor([t], dtype=torch.float32)
            ).flatten()[0].item()

        exact = sigma_of(937.5)
        truncated = sigma_of(937.0)
        self.assertAlmostEqual(exact, 0.9375, places=4)
        self.assertNotAlmostEqual(exact, truncated, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
