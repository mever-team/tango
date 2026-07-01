# TANGO — Project Page

Project page for the ECCV 2026 paper **"Test-Time Noise Guided Adaptation for Realistic Autoregressive Video Generation" (TANGO)**.

🔗 Live page: https://mever-team.github.io/tango/

> This is currently a **placeholder** page (title, authors, abstract, citation). Qualitative results, method overview, and video comparisons will be added.

## Local preview

It is a static site — open `index.html` directly, or serve it:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

## Deploying to GitHub Pages

1. Create a repository named `tango` under the `mever-team` organization and push this directory to it.
2. In **Settings → Pages**, set **Source: Deploy from a branch**, branch `main`, folder `/ (root)`.
3. The site will be published at `https://mever-team.github.io/tango/`.

The `.nojekyll` file is included so GitHub Pages serves the `static/` assets without Jekyll processing.

## Structure

```
index.html          # the page
static/css/         # Bulma + template styles
static/js/          # template scripts (carousel, slider, helpers)
static/images/      # favicon (add figures/social_preview.png here later)
```

## Attribution

Built with the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), adapted from the [Nerfies](https://nerfies.github.io) project page. Licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).
