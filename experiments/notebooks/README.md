# Notebooks for the experiments added after the first release

Kaggle notebooks (run top to bottom; outputs go to `results/<model>/`). Same harness, prompts, gates and scoring as `../e6_frontier_api.py`.

| Notebook | Experiments | Output files in `data/` |
|---|---|---|
| `v9_frontier_and_nemotron.ipynb` | E3′, E7 (frontier models, Nemotron-3-Super), T=0.7 probes | `*_v9*`, `nemotron-3-super_v9free*` |
| `e8_real_browser.ipynb` | E8 | `*_e8*_episodes.csv` |
| `e9_indicator.ipynb` | E9 | `nemotron-3-super_e9ind_*` |
| `e10_robustness_nemotron.ipynb` | R′, Nemotron ICP | `nemotron-3-super_e10_*` |
| `q3_qwen3.ipynb` | R (Qwen3-8B on E3–E9) | `qwen3-8b_q3_*` |
| `icp.ipynb` | ICP (local models) | `*_icp_*` |
| `e11_policy_prompt_and_guards.ipynb` | E11 | `*_e11_*`, `e11_guard_*` |
| `e12_adaptive_attacker.ipynb` | E12 | `*_e12_*` |
