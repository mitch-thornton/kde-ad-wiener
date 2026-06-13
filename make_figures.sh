#!/usr/bin/env bash
# Regenerate the paper's figures from the bundled scripts and data.
# This repository contains code and data only; it does NOT build the document.
# The compiled paper is distributed separately as KDE_AD_Wiener_arXiv-v1.pdf.
#
# Usage:  ./make_figures.sh
# Output: figures/*.pdf
#
# Requires: numpy, scipy, matplotlib  (see requirements.txt)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
mkdir -p figures

echo "[1/9] examples, hard densities, heaped-data robustness ..."
python3 scripts/ad_kde_v30.py        | tail -1   # fig_kde_examples, fig_kde_hard, fig_kde_heaped
echo "[2/9] bandwidth-selection mechanism and sweep ..."
python3 scripts/ad_bw_core_v1.py     | tail -1   # fig_mechanism, fig_bw_sweep
echo "[3/9] selector comparison ..."
python3 scripts/ad_bw_compare_v1.py  | tail -1   # fig_b99_compare
echo "[4/9] signal-bandwidth generality (chirp) ..."
python3 scripts/ad_bw_chirp_v1.py    | tail -1   # fig_chirp
echo "[5/9] deconvolution under known measurement error ..."
python3 scripts/exp_deconv_v13.py    | tail -1   # fig_deconv
echo "[6/9] mixed-mode, superposition, and synthetic-data battery ..."
python3 scripts/exp_datagen_v30.py   | tail -1   # fig_superpose, fig_mixedmode, fig_alternating, fig_datagen, fig_datagen_sweep
echo "[7/9] financial tail-risk illustration ..."
python3 scripts/exp_financial_v9.py  | tail -1   # fig_financial
echo "[8/9] benchmark average ranks ..."
python3 scripts/make_benchmark_fig.py | tail -1  # fig_benchmark (ranks recomputed from the published ISE matrix)
echo "[9/9] CRSP distribution and tail figures (from bundled results/) ..."
python3 scripts/make_crsp_figs.py    | tail -1   # fig_crsp_dist, fig_crsp_tail

echo
echo "Random-beacon table (from the vendored sample in data/):"
python3 scripts/exp_trng_v30.py      | tail -3   || true

cat <<'NOTE'

Done. Figures written to figures/.

Not regenerated here (they require public datasets you must download yourself; see DATA.md):
  - fig_cern_dimuon   : scripts/exp_cern_dimuon_v17.py   (CMS dimuon, CERN Open Data, CC0)
  - fig_sdss_redshift : scripts/exp_sdss_redshift_v16.py (SDSS DR18, SkyServer)
  - fig_cyber         : scripts/exp_cyber_unsw_v27.py    (UNSW-NB15)
  - fig_heaping_*     : scripts/exp_nhanes_heaping_v14.py --data-dir <NHANES_DIR> (U.S. CDC NHANES)
The CRSP figures above are regenerated from the bundled derived results in results/, so the
proprietary raw CRSP data is not needed.
NOTE
