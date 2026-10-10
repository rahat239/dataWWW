# JobLure: knowing–doing gap of LLM job-search agents under employment scams

Anonymous artifact for the WWW 2027 submission *"Knowing Is Not Doing: LLM Job-Search Agents Recognize Employment Scams Yet Hand Over Identity, Bank Details, and Fees"*.

It contains the four experiment scripts, every agent trial log (3,264 trials), the standalone "knowing" probes, and the analysis that regenerates all numbers and figures in the paper.

## Layout

```
experiments/   one script per experiment (Jupyter-style "# %%" cells; run as a notebook or as a script)
  e1_instructions.py              E1: factorial PII requests x {neutral, "assess scam risk first", "never share"}
  e2_request_types_defenses.py    E2: chat / own-domain portal / look-alike portal / $89 fee x {neutral, generic safety, self-check gate}
  e3_gates.py                     E3: 4 scam + 3 legitimate scenarios x {none, msg, ctx, policy, hybrid} gates
  e4_ablation_paraphrase_scale.py E4: payment-permission ablation, hybrid+ gate, paraphrase probes, larger models (4-bit)
analysis/
  paper_stats.py                  all paper numbers -> results/stats.json, figures -> results/figures/
  e1_summary.py ... e4_summary.py per-experiment summary tables (markdown)
data/                             trial logs and probes (see below)
results/                          stats.json and figure PDFs as used in the paper
```

## Reproducing the paper's numbers (no GPU needed)

```
pip install -r requirements.txt
python analysis/paper_stats.py data results/figures      # regenerates results/stats.json and the figures
python analysis/e3_summary.py data out                   # optional per-experiment tables
```

`results/stats.json` in this archive is byte-for-byte what the paper reports.

## Re-running the experiments

All runs used two NVIDIA T4 GPUs, HuggingFace Transformers, and local open-weight models only (no external APIs):
Gemma-2-9B-it, Phi-3.5-mini-instruct, Qwen2.5-7B-Instruct (fp16) and Qwen2.5-14B-Instruct (4-bit NF4, via bitsandbytes).

1. Download the EMSCAD corpus (`fake_job_postings.csv`, "Employment Scam Aegean Dataset", Vidros et al., Future Internet 2017) and place it anywhere under the working directory; the scripts find it automatically.
2. Make one model available locally and set `CONFIG["model_hint"]` (a substring of the model path) or `CONFIG["model_path"]`.
3. For E4 set `CONFIG["conditions"]` (e.g. `["neutral", "neutral_noauth", "hybrid_plus"]`) and, for 14B, `CONFIG["quant"] = "4bit"`.
4. Run the script top to bottom. Outputs go to `results/<model-tag>/`.

Setting `CONFIG["backend"] = "mock"` runs the whole pipeline offline with a random stand-in model, to check the environment without a GPU (mock results are meaningless).

Fixed settings: seed 42; 60 fraudulent + 180 legitimate postings matched on entry-level share; temperature 0.7, top-p 0.95 for the agent; temperature 0 for probes and gates; 4 (E1–E2) or 5 (E3–E4) agent steps after the recruiter message.

## Data files

| File pattern | Experiment | Unit |
|---|---|---|
| `{model}_controlled.csv` | E1 | one row per agent trial |
| `{model}_v4.csv`, `{model}_v4_summary.json` | E2 | trials (Gemma, Phi); summary only for Qwen2.5-7B |
| `{model}_v5.csv`, `phi-3.5-mini_v5_hybrid.csv` | E3 | trials (Phi's hybrid gate was run separately) |
| `{model}_knowing_v5.csv`, `qwen2.5-14b_knowing_v6big.csv` | E3/E4 | standalone probes (message only / with application record) |
| `{model}_v6.csv`, `qwen2.5-14b_v6big.csv` | E4 | trials |
| `{model}_v6_paraphrase_knowing.csv` | E4 | paraphrase probes; `family` x `para` x `title` pairs of `variant` own / lookalike |

Key trial columns (E3/E4): `condition`, `scenario`, `truth` (scam/legit), `harm` (canary sent to the adversary or payment executed), `attempted`, `paid`, `legit_completed` (data reached the real employer), `blocked`, `gate_calls` (JSON: tool, decision, stage = origin / provenance / llm, raw gate output), `warns_user`, `gave_timeslots`, `reply`, `forms`, `payments`, `report`, `actions` (raw model outputs).

## Safety and privacy

All personal identifiers are synthetic canaries (`CNRY-...`); the user profile ("Rahim Uddin") is fictional. Employers in E3/E4 and all scam domains are fictional, and no real website was contacted. Posting text comes from the public EMSCAD research corpus.

## License

Code: MIT (see LICENSE). Trial logs and probes: CC BY 4.0. EMSCAD is not redistributed here; obtain it from its original source.
