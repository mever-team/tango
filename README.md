# TANGO — Project Page

Project page for the ECCV 2026 paper **“Test-Time Noise Guided Adaptation for Realistic Autoregressive Video Generation” (TANGO)**.

🔗 Live page: https://mever-team.github.io/tango/ · 📄 Paper: https://arxiv.org/abs/2607.15849

A fully custom, dependency-free static site (no frameworks, no CDNs, self-hosted fonts). Dark theme by default with a light-theme toggle (persisted in `localStorage`; `?theme=light` also works).

## Local preview

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

Opening `index.html` directly also works, except the self-hosted fonts, which some browsers block on `file://`.

## Structure

```
index.html            # the whole page: content, inline SVG charts + manifold diagram
static/css/style.css  # design tokens (dark/light), layout, components
static/js/main.js     # theme toggle, filmstrip scrubbers, synced sliders, tabs,
                      # chart tooltips, BibTeX copy, scrollspy, reveals
static/fonts/         # self-hosted variable Inter, Source Serif 4 (600), CM Caligraphic (math glyphs)
static/images/
  gallery/<sample>/   # per-sample video frames f1..f7.jpg (+ input.jpg for I2V/V2V)
  compare/<scene>/    # 3-method × 4-segment comparison frames (from paper Fig. 3)
  failures/           # failure-case frames
  arch/               # frames embedded in the architecture diagram (from paper Fig. 2)
  strips/             # frames from the paper's overview figure (currently unused spare)
  social_preview.png  # 1200×630 Open Graph card
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{karageorgiou2026tango,
  title     = {Test-Time Noise Guided Adaptation for Realistic Autoregressive Video Generation},
  author    = {Karageorgiou, Dimitrios and Papadopoulos, Symeon and Kompatsiaris, Ioannis and Gavves, Efstratios},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgments

This work was supported by the Horizon Europe projects [ELIAS](https://elias-ai.eu/) (grant no. 101120237) and [ELLIOT](https://elliot-ai.eu/) (grant no. 101214398).
The computational resources were granted with the support of [GRNET](https://grnet.gr/en/).

## License

Website content: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). © 2026 the authors.
