# Additional baseline methods

## What this version tests
One extra changepoint-detection baseline that is methodologically distinct from the TCPD-paper competitors and is run via an official, well-maintained package (`hmmlearn`).

The other baselines originally considered here (BS-RSS, PELT-RSS, Bayesian-CPD, Kernel-CPD, Maulik-ML) were dropped during revision:

- **Kernel-CPD** duplicates the TCPD competitor `KernelCPD`.
- **BS-RSS, PELT-RSS, Bayesian-CPD, Maulik-ML** either overlap conceptually with existing competitors or lack an official implementation.

## Settings
- **Script:** `run_hmm_regime_tcpd.py`
- **Environment:** `revision/venv_cpd` (Python, with `hmmlearn`)
- **Standardization:** z-score standardization before fitting the GaussianHMM.
- **Method:** `hmmlearn.hmm.GaussianHMM` with diagonal covariance.
- **Default run:** `K = 3` states, `min_seg_len = 10`.
- **Oracle run:** select the best `K` from `{2, 3, 4, 5}` by per-dataset F1.
- **Changepoints:** declared at Viterbi state transitions, then filtered to a minimum gap of 10.
- **Evaluation metric:** TCPD-style F1 / Covering with margin 5, averaged over annotator CP sets.

## Result files
- `benchmark_tcpd_hmm_regime_default.json`
- `benchmark_tcpd_hmm_regime_oracle.json`

## Legacy files
The old from-scratch Julia script and its JSONs remain in this folder for reference but are no longer used to build the manuscript tables/figures:

- `benchmark_tcpd_additional_baselines.jl`
- `benchmark_tcpd_additional_baselines_default.json`
- `benchmark_tcpd_additional_baselines_oracle.json`
