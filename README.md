<div align="center">

<img src="assets/logo.svg" alt="" width="72" height="72">

# TANGO

**Test-Time Noise Guided Adaptation for Realistic Autoregressive Video Generation**

Dimitrios Karageorgiou<sup>1,2</sup> · Symeon Papadopoulos<sup>1</sup> · Ioannis Kompatsiaris<sup>1</sup> · Efstratios Gavves<sup>2</sup>

<sup>1</sup> Information Technologies Institute, CERTH · <sup>2</sup> University of Amsterdam

ECCV 2026

[![arXiv](https://img.shields.io/badge/arXiv-2607.15849-b31b1b?style=flat-square)](https://arxiv.org/abs/2607.15849)
[![Project page](https://img.shields.io/badge/project%20page-mever--team.github.io%2Ftango-1fa694?style=flat-square)](https://mever-team.github.io/tango/)

</div>

> The code has not been released yet. Watch this repository to be notified.

Autoregressive video diffusion models eventually collapse. Prior works aim to keep each frame on the
manifold of real ones, but a trajectory whose every frame looks right can still reach a **terminal
point**: a state in the manifold of real videos that the model lacks the knowledge to continue.
TANGO detects terminal points at test time from the model's own noise predictions. Then, test-time
adaptation is employed to steer the model away from terminal points, effectively trading
inference-time compute for improved performance.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/architecture-dark.png">
  <img src="assets/architecture-light.png" alt="Test-time adaptation loop: the adapted model proposes a candidate next frame, the frozen model predicts one step beyond it, and the deviation of that look-ahead residual from isotropic Gaussian noise drives the update.">
</picture>

## Citation

```bibtex
@inproceedings{karageorgiou2026tango,
  title     = {Test-Time Noise Guided Adaptation for Realistic Autoregressive Video Generation},
  author    = {Karageorgiou, Dimitrios and Papadopoulos, Symeon and Kompatsiaris, Ioannis and Gavves, Efstratios},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgments

Supported by the Horizon Europe projects [ELIAS](https://elias-ai.eu/) (grant no. 101120237) and
[ELLIOT](https://elliot-ai.eu/) (grant no. 101214398). Computational resources were granted with the
support of [GRNET](https://grnet.gr/en/).
