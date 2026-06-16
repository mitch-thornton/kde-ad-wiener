# kde-ad-wiener

Code and data to reproduce the figures of the [arXiv paper](https://arxiv.org/abs/2606.15450):

**"Kernel Density Estimation by Spectral Decomposition: Data-Driven Tapering and Superposition"**

The compiled paper is distributed separately here as `KDE_AD_Wiener_arXiv-v1.pdf`. This repository contains
the scripts and the redistributable derived data only; it does not contain the manuscript source and
does not build the document.

## What the paper does
Bandwidth selection and density estimation are read in the characteristic-function domain: the cyclic
group-averaged covariance of the binned data has the squared empirical characteristic function as its
spectrum, and the bandwidth is the cutoff where that spectrum meets the 1/n sampling-noise floor. From
this follow an automatic bandwidth selector, an adaptive per-frequency Wiener taper computed from the
data, deconvolution under known measurement error, a piecewise mixed-mode estimator, a smooth-base /
band-limited-residual superposition (the default), and a data-driven noise floor that replaces the
assumed 1/n floor and stays robust on heaped and rounded data.

## Regenerate the figures
    pip install -r requirements.txt
    ./make_figures.sh

This regenerates every figure whose inputs are bundled here, into `figures/`:
the examples, hard-density, and heaped-data figures; the bandwidth-selection mechanism, sweep, and
comparison; the chirp generality figure; the deconvolution figure; the mixed-mode, superposition, and
synthetic-data battery figures; the financial tail-risk figure; the benchmark average-rank figure; and
the CRSP distribution and tail figures (from the derived results in `results/`). It also prints the
random-beacon table from the vendored sample in `data/`.

Four figures depend on public datasets you must download yourself (CMS dimuon, SDSS DR18, UNSW-NB15,
and NHANES); their scripts are included and `DATA.md` gives the sources and commands.

## Layout
    scripts/        figure-producing scripts and shared modules
    results/        derived per-series CRSP results (JSON) that reproduce the CRSP figures
    data/           vendored random-beacon sample used by the beacon table
    figures/        output directory, populated by make_figures.sh
    make_figures.sh one-command figure regeneration
    requirements.txt
    DATA.md         data sources, provenance, and per-script reproduction
    LICENSE         MIT (code)
    PATENTS.md      patent notice

## License
Code is released under the MIT License (`LICENSE`). See `PATENTS.md` for the patent notice. The
benchmark figure recomputes its average ranks directly from the per-density integrated-squared-error
matrix reported in the paper, so the figure and the paper's tables are guaranteed consistent.
