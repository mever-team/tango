"""Regression tests for run reproducibility and cross-arm pairing.

Background
----------
Two defects made every TTO run non-reproducible and every pair of arms
incomparable:

1. `--seed` was declared by all three entry scripts and **never read**, while
   every config sets `seed: 0`, which `Trainer.__init__` turned into
   `torch.randint(...)` from an unseeded generator. So each launch used a
   different, unlogged master seed and no run reproduced even itself.

2. The per-video trajectory noise was drawn from the **global** generator inside
   the video loop. Anything that changes how many random numbers are consumed
   (TTO epochs, LoRA resets, per-epoch look-ahead draws) shifts the stream, so
   every *subsequent* video gets different noise. Two arms differing only in
   their TTO settings therefore produce different videos for reasons unrelated
   to the treatment.

The fix: honour `--seed`, use a literal seed unless a negative one explicitly
requests randomisation (and log it either way), and draw the per-video noise from
a generator seeded by `(master_seed, dataset idx)`.

Run:  python -m unittest discover -s tests -v
"""

import ast
import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENTRY_SCRIPTS = (
    "inference_tto.py",
    "inference_tto_gaussian.py",
    "inference_tto_gaussian_optimized.py",
)


def _read(rel):
    with open(os.path.join(REPO_ROOT, rel)) as fh:
        return fh.read()


class TestSeedForVideo(unittest.TestCase):
    """The per-video seed derivation itself."""

    @staticmethod
    def _fn():
        import sys
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        # utils.misc imports torch; fall back to an AST-extracted copy when
        # torch is unavailable so this class still runs.
        try:
            from utils.misc import seed_for_video
            return seed_for_video
        except Exception:
            src = _read("utils/misc.py")
            tree = ast.parse(src)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "seed_for_video":
                    ns = {}
                    exec(compile(ast.Module([node], []), "<seed_for_video>", "exec"), ns)
                    return ns["seed_for_video"]
            raise unittest.SkipTest("seed_for_video not found")

    def test_is_deterministic(self):
        f = self._fn()
        self.assertEqual(f(1234, 7), f(1234, 7))

    def test_distinct_videos_get_distinct_seeds(self):
        f = self._fn()
        seeds = [f(1234, i) for i in range(2000)]
        self.assertEqual(len(seeds), len(set(seeds)), "per-video seeds must not collide")

    def test_distinct_master_seeds_give_distinct_streams(self):
        f = self._fn()
        self.assertNotEqual(f(1, 0), f(2, 0))
        # and the whole per-video sequence differs
        self.assertNotEqual([f(1, i) for i in range(50)], [f(2, i) for i in range(50)])

    def test_result_is_a_valid_torch_seed(self):
        f = self._fn()
        for master in (0, 1, 1234, 2 ** 31 - 2):
            for idx in (0, 1, 182, 999):
                s = f(master, idx)
                self.assertIsInstance(s, int)
                self.assertGreaterEqual(s, 0)
                self.assertLess(s, 2 ** 63 - 1)


class TestEntryScriptsHonourSeed(unittest.TestCase):
    """`--seed` must actually reach the config, and the noise must be isolated."""

    def test_seed_argument_is_read(self):
        for rel in ENTRY_SCRIPTS:
            src = _read(rel)
            self.assertIn('"--seed"', src, f"{rel} must declare --seed")
            self.assertIn("args.seed", src,
                          f"{rel} declares --seed but never reads it — the flag is inert")
            self.assertIn("config.seed = args.seed", src,
                          f"{rel} must assign --seed into the config before the Trainer is built")

    def test_seed_is_wired_before_trainer_construction(self):
        """Assigning after `Trainer(config)` would be too late: the Trainer consumes it."""
        for rel in ENTRY_SCRIPTS:
            src = _read(rel)
            wire = src.index("config.seed = args.seed")
            build = src.index("Trainer(config)")
            self.assertLess(wire, build,
                            f"{rel}: --seed must be wired before Trainer(config)")

    def test_per_video_noise_uses_an_isolated_generator(self):
        for rel in ENTRY_SCRIPTS:
            src = _read(rel)
            self.assertIn("seed_for_video(", src,
                          f"{rel} must derive a per-video seed")
            tree = ast.parse(src)
            offenders = []
            for node in ast.walk(tree):
                # every `sampled_noise = torch.randn(...)` must pass a generator
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if "sampled_noise" not in names or node.value is None:
                    continue
                call = node.value
                if not (isinstance(call, ast.Call)
                        and (ast.get_source_segment(src, call.func) or "").endswith("randn")):
                    continue
                if not any(kw.arg == "generator" for kw in call.keywords):
                    offenders.append(f"{rel}:{node.lineno}")
            self.assertEqual(
                offenders, [],
                "The per-video trajectory noise must come from a generator seeded by "
                "(master_seed, idx); drawing it from the global RNG makes it depend on "
                "how many draws earlier videos consumed, which breaks cross-arm "
                "pairing.\n" + "\n".join(offenders),
            )


class TestTrainerSeedIsDeterministic(unittest.TestCase):
    """The master seed must be deterministic by default and always logged."""

    def test_seed_zero_is_not_silently_randomised(self):
        src = _read("tto/trainer.py")
        self.assertNotIn(
            "if config.seed == 0:", src,
            "`seed: 0` (which every config sets) must not trigger a random master "
            "seed — that is what made runs non-reproducible.",
        )

    def test_random_seed_is_logged(self):
        """If a random seed is drawn, it must be printed so the run stays reproducible."""
        src = _read("tto/trainer.py")
        tree = ast.parse(src)
        randint_lines = [
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and (ast.get_source_segment(src, n.func) or "").endswith("randint")
        ]
        if not randint_lines:
            return  # no randomisation path at all is also acceptable
        lines = src.split("\n")
        for ln in randint_lines:
            window = "\n".join(lines[max(0, ln - 3):ln + 6])
            self.assertIn(
                "print", window,
                f"tto/trainer.py:{ln}: a randomly drawn master seed must be logged",
            )


class TestGeneratorIsolation(unittest.TestCase):
    """The property the fix exists for, checked against torch."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"torch unavailable: {exc}")

    def _noise(self, seed, prior_draws):
        """Draw 'trajectory noise' after consuming `prior_draws` globals."""
        import torch
        torch.manual_seed(0)
        for _ in range(prior_draws):          # simulate a differing TTO config
            torch.randn(8)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        return torch.randn([2, 3, 4], generator=gen)

    def test_noise_is_independent_of_prior_global_consumption(self):
        """This is the pairing property: two arms that consumed a different number
        of random numbers must still get identical trajectory noise."""
        import torch
        a = self._noise(12345, prior_draws=0)
        b = self._noise(12345, prior_draws=997)
        self.assertTrue(torch.equal(a, b),
                        "trajectory noise must not depend on prior global RNG use")

    def test_different_videos_get_different_noise(self):
        import torch
        import sys
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from utils.misc import seed_for_video
        a = self._noise(seed_for_video(1234, 0), 0)
        b = self._noise(seed_for_video(1234, 1), 0)
        self.assertFalse(torch.equal(a, b))

    def test_same_video_same_seed_reproduces(self):
        import torch
        import sys
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from utils.misc import seed_for_video
        s = seed_for_video(1234, 42)
        self.assertTrue(torch.equal(self._noise(s, 0), self._noise(s, 13)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
