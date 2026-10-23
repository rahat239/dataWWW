# Experiments added after the first release

The Kaggle working notebooks for the runs below are not included here. They reuse the harness, prompts, gates and scoring in `../e6_frontier_api.py` (E9b, E5b and the HTTP E13 run are in `../e9b_e5b_e13_kaggle.ipynb`, the HTTPS E13 run in `../e13_https_kaggle.ipynb`). Every trial and episode log is in `data/`, and `analysis/new_experiments.py` recomputes every number the paper reports for these experiments.

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
| E13 (browser-use) | `*_e13https_episodes.csv` (HTTPS, reported); `nemotron-3-super_e13_episodes.csv`, `gpt-6.1-sol_e13_episodes.csv` (earlier HTTP run) |
| E14, E15 (inside the agent; fine-tune) | `e14_e15/` (notebooks: `../e14_inside_the_gap_kaggle.ipynb`, `../e15_probes_and_finetune_kaggle.ipynb`) |
