# JobLure: knowing–doing gap of LLM job-search agents under employment scams

Anonymous artifact for the WWW 2027 submission *"Knowing Is Not Doing: Open-Weight LLM Job-Search Agents Act on Employment Scams Their Own Models Flag"*.

It contains the experiment code, the anonymized real-message corpus, every agent trial and browser-episode log behind the paper's 13,664 trials (E1–E13, R, R′, ICP), the standalone "knowing" probes, and the analysis that regenerates all numbers and figures in the paper.

## Layout

```
experiments/   one script per experiment (Jupyter-style "# %%" cells; run as a notebook or as a script)
  e1_instructions.py              E1: factorial PII requests x {neutral, "assess scam risk first", "never share"}
  e2_request_types_defenses.py    E2: chat / own-domain portal / look-alike portal / $89 fee x {neutral, generic safety, self-check gate}
  e3_gates.py                     E3: 4 scam + 3 legitimate scenarios x {none, msg, ctx, policy, hybrid} gates
  e4_ablation_paraphrase_scale.py E4: payment-permission ablation, hybrid+ gate, paraphrase probes, larger models (4-bit)
  e6_frontier_api.py              E6: frontier models through the OpenRouter API (needs OPENROUTER_API_KEY)
  e5_real_messages_reps.py        E5: 45 real scam + 10 real legitimate messages; also repeats E3/E4 main conditions with a new seed
  e9b_e5b_e13_kaggle.ipynb        E9b (false video-interview claim vs. the FBI rule), E5b (new real messages; template in
                                  data/real_messages/), E13 (browser-use in Chromium behind a local proxy that hosts the
                                  sites and runs the hybrid+ guard on every POST). Built on e6_frontier_api.py.
  e13_https_kaggle.ipynb          E13 as reported: the same harness with every site served over HTTPS (proxy terminates TLS)
  notebooks/README.md             which data/ files hold E3'/E7, E8, E9, E10 (R'), E11, E12, ICP and R (same harness as e6)
analysis/
  paper_stats.py                  all paper numbers -> results/stats.json, figures -> results/figures/
  reanalysis.py                   sensitivity/specificity, E1/E2 clear-scam cells, E5 conditional and payee split,
                                  message-level cluster bootstrap, Holm correction -> results/stats_review.json
  rescore_v7.py                   strict off-platform harm for E5 (only messages to scammer-supplied contacts count)
  new_experiments.py              every number for ICP, E3'/E7 (Nemotron), E8, E9, E9b, E11, E12, E13, R and R'
  e1_summary.py ... e4_summary.py per-experiment summary tables (markdown)
data/                             trial logs and probes (see below); data/real_messages/ = E5 corpus with sources
results/                          stats.json and figure PDFs as used in the paper
```

## Reproducing the paper's numbers (no GPU needed)

```
pip install -r requirements.txt
cd analysis
python paper_stats.py ../data ../results/figures                 # regenerates results/stats.json and the figures
python reanalysis.py ../data ../results/stats_review.json        # review statistics (Table 2, E5 splits, CIs, Holm)
python e7_analysis.py ../data ../results/stats_e7.json           # E7, Claude re-run, T=0.7 probes
python new_experiments.py ../data                                # ICP, E7-E13, R, R' (prints one line per reported number)
python analysis/e3_summary.py data out                   # optional per-experiment tables
```

`results/stats.json` and `results/stats_review.json` in this archive are what the paper reports.

## Re-running the experiments

Local open-weight runs used two NVIDIA T4 GPUs and HuggingFace Transformers: Gemma-2-9B-it, Phi-3.5-mini-instruct, Qwen2.5-7B-Instruct, Granite-3.1-8B-Instruct, Qwen3-8B (fp16) and Qwen2.5-14B-Instruct (4-bit NF4, via bitsandbytes). Nemotron-3-Super and the frontier models were called through the OpenRouter API (set `OPENROUTER_API_KEY`); E8 and E13 drive headless Chromium.

1. Download the EMSCAD corpus (`fake_job_postings.csv`, "Employment Scam Aegean Dataset", Vidros et al., Future Internet 2017) and place it anywhere under the working directory; the scripts find it automatically.
2. Make one model available locally and set `CONFIG["model_hint"]` (a substring of the model path) or `CONFIG["model_path"]`.
3. For E4 set `CONFIG["conditions"]` (e.g. `["neutral", "neutral_noauth", "hybrid_plus"]`) and, for 14B, `CONFIG["quant"] = "4bit"`.
4. Run the script top to bottom. Outputs go to `results/<model-tag>/`.

Setting `CONFIG["backend"] = "mock"` runs the whole pipeline offline with a random stand-in model, to check the environment without a GPU (mock results are meaningless).

Fixed settings: seed 42; 60 fraudulent + 180 legitimate postings matched on entry-level share; temperature 0.7, top-p 0.95 for the agent; temperature 0 for probes and gates; 4 (E1–E2) or 5 (E3–E5) agent steps after the recruiter message.

## Data files

| File pattern | Experiment | Unit |
|---|---|---|
| `{model}_controlled.csv` | E1 | one row per agent trial |
| `{model}_v4.csv`, `{model}_v4_summary.json` | E2 | trials (Gemma, Phi); summary only for Qwen2.5-7B |
| `{model}_v5.csv`, `phi-3.5-mini_v5_hybrid.csv` | E3 | trials (Phi's hybrid gate was run separately) |
| `{model}_knowing_v5.csv`, `qwen2.5-14b_knowing_v6big.csv` | E3/E4 | standalone probes (message only / with application record) |
| `{model}_v6.csv`, `qwen2.5-14b_v6big.csv` | E4 | trials |
| `{model}_v6_paraphrase_knowing.csv` | E4 | paraphrase probes; `family` x `para` x `title` pairs of `variant` own / lookalike |
| `{model}_v7_trials.csv` | E3/E4 | repeated main conditions (new trial seed; all conditions for Granite) |
| `granite-3.1-8b_v7_knowing.csv`, `granite-3.1-8b_v7_paraphrase_knowing.csv` | E3/E4 | Granite probes |
| `{model}_v7_real_trials.csv`, `{model}_v7_real_knowing.csv` | E5 | real-message trials and probes |
| `gemma-2-9b_v7fix_real_trials.csv` | E5 | Gemma hybrid+ re-run after the own-contact origin fix (replaces those rows) |
| `{model}_v8_*.csv` | E6 | frontier-model probes and trials (Claude's probes: re-run `_v8b_*`) |
| `{frontier}_v9_trials.csv`, `claude-sonnet-5.5_v9_real_*.csv` | E6 | Claude re-run with a 4,096-token budget |
| `{frontier}_v9_adaptive_trials.csv`, `_v9_adaptive_knowing.csv` | E7 | fraudulent employer on its own domain after a chat-only interview |
| `{frontier}_v9_knowing_t07.csv` | E3/E7 | probes re-sampled 5x at T=0.7 |
| `real_messages/real_messages.csv` | E5 | 55 anonymized messages with label, ask type, channel and source URL |
| `nemotron-3-super_v9free_*`, `_v9freeprobes_*` | E3′, E7 | Nemotron-3-Super trials (none / hyb+), E7 own-domain scenarios, probes |
| `nemotron-3-super_e8{,run2,run3}_episodes.csv`, `qwen3-8b_e8qwen3_episodes.csv` | E8 | real-browser episodes (three Nemotron runs, 3 episodes per cell each) |
| `nemotron-3-super_e9ind_*` | E9 | FBI chat-interview rule as policy rule 4 (E3, E7 and E5 scenarios) |
| `{model}_e9b_trials.csv`, `_e9b_knowing.csv` | E9b | E7 scenarios with / without a false video-interview claim; scenario = `<base>|claim` or `<base>|noclaim` |
| `qwen3-8b_q3_*` | R | Qwen3-8B: E3, E7, probes, real messages (79), label-free record (`_real_neutral_*`), ATS hosts, other prompts |
| `nemotron-3-super_e10_*` | R′, ICP | Nemotron-3-Super: label-free record, ATS hosts, other prompts, neutral probe wording, in-context probes, T=0.7 probes |
| `{model}_icp_*` | ICP | in-context probes (`icp_t0` = verdict inside the agent context), T=0.7 probes, money-mentioning legitimate messages |
| `{model}_e11_policy_prompt_trials.csv` | E11 | gate policy (and rule 4) in the agent's system prompt |
| `e11_guard_*.csv` | E11 | guard models judged offline on the same 130 proposed actions (`block*` columns) |
| `{model}_e12_attack.csv`, `_ablation_trials.csv`, `_transfer_trials.csv` | E12 | message rewriting against the gate (log per round), instruction/tool ablations, transfer |
| `nemotron-3-super_e13_episodes.csv` | E13 | browser-use episodes; `posts` = every POST seen by the guard with its decision and stage |
| `gpt-6.1-sol_e13_episodes.csv` | E13 | same, GPT-6.1 Sol (3 episodes per cell); this and the Nemotron file above are the earlier run over plain HTTP |
| `*_e13https_episodes.csv` | E13 | the run reported in the main text: all sites over HTTPS, Nemotron-3-Super, GPT-6.1 Sol and Gemini 3.1 Pro, 5 episodes per cell. Gemini's hybrid_plus look-alike, own-domain and payroll episodes come from a re-run after an OpenRouter credit stop had emptied the first attempt |
| `pilots_and_failed_runs/` | – | pilots and failed runs (e.g. Claude through browser-use: no valid action); not used for any number |


Key trial columns (E3/E4): `condition`, `scenario`, `truth` (scam/legit), `harm` (canary sent to the adversary or payment executed), `attempted`, `paid`, `legit_completed` (data reached the real employer), `blocked`, `gate_calls` (JSON: tool, decision, stage = origin / provenance / llm, raw gate output), `warns_user`, `gave_timeslots`, `reply`, `forms`, `payments`, `report`, `actions` (raw model outputs).

## Safety and privacy

All personal identifiers are synthetic canaries (`CNRY-...`); the user profile is fictional. For the release, the profile's display name, e-mail and phone number used during the runs were replaced throughout code and logs by the pseudonym "Alex Morgan", `alex.morgan.jobs@example.org` and `+1-555-0100-000000` (case-preserving string substitution, also inside model outputs), and city/country names and time-zone offsets that models derived from the profile are redacted as `[CITY]`/`[COUNTRY]`/`[OFFSET]`. The same substitution was applied to the runs added after the first release. Canary values, scammer contacts and all other fields are unchanged, and every number in the paper regenerates exactly from the released logs. Employers in E3/E4 and all scam domains are fictional, and no real website was contacted (in E8 and E13 the sites are self-hosted; the E13 proxy refuses every other host). Posting text comes from the public EMSCAD research corpus.

## License

Code: MIT (see LICENSE). Trial logs and probes: CC BY 4.0. EMSCAD is not redistributed here; obtain it from its original source.
