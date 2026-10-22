# Experiments added after the first release

The Kaggle working notebooks for the runs below are not included here. They reuse the harness, prompts, gates and scoring in `../e6_frontier_api.py` (E9b, E5b and E13 are in `../e9b_e5b_e13_kaggle.ipynb`, which also contains the browser-use setup for E13). Every trial and episode log is in `data/`, and `analysis/new_experiments.py` recomputes every number the paper reports for these experiments.

| Experiments | Trial / episode logs in `data/` |
|---|---|
| E3′, E7 (frontier models, Nemotron-3-Super), T=0.7 probes | `*_v9*`, `nemotron-3-super_v9free*` |
| E8 (real browser) | `*_e8*_episodes.csv` |
| E9 (FBI indicator in the policy) | `nemotron-3-super_e9ind_*` |
| E9b (false video-interview claim) | `*_e9b_trials.csv` |
| R′, Nemotron ICP | `nemotron-3-super_e10_*` |
| R (Qwen3-8B on E3–E9) | `qwen3-8b_q3_*` |
| ICP (local models) | `*_icp_*` |
| E11 (policy in the prompt; guard models) | `*_e11_*`, `e11_guard_*` |
| E12 (gate-aware attacker, ablations) | `*_e12_*` |
| E13 (browser-use) | `nemotron-3-super_e13_episodes.csv` |
