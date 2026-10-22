"""Recompute the paper's numbers for the experiments added after the first release:
ICP, E3'/E7 (Nemotron), E8, E9, E9b, E11, E12, E13 (Nemotron, GPT-6.1), and R/R' (Qwen3-8B, Nemotron).

    python analysis/new_experiments.py data

Prints one line per reported number: section, quantity, value. Needs pandas only.
"""
import sys, os, json
from math import comb
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else "data"
NEM, Q3 = "nemotron-3-super", "qwen3-8b"
OPEN6 = ["gemma-2-9b", "phi-3.5-mini", "qwen2.5-7b", "qwen2.5-14b", "granite-3.1-8b", Q3]


def r(name):
    return pd.read_csv(os.path.join(D, name))


def out(sec, what, val):
    print(f"{sec:6s} {what:70s} {val}")


def frac(s):
    s = pd.Series(s)
    return f"{int(s.sum())}/{len(s)}"


def fisher(a, n1, b, n2):
    K, N = a + b, n1 + n2
    p = lambda x: comb(n1, x) * comb(n2, K - x) / comb(N, K)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, K - n2), min(n1, K) + 1) if p(x) <= p0 * (1 + 1e-9))


ONBOARD = ["payroll_offer", "background_offer"]

# ---------------- ICP: in-context probes (seven open models) -------------------------------------------------
icp = pd.concat([r(f"{m}_icp_trials.csv").assign(model=m) for m in OPEN6 if m != Q3] +
                [r(f"{Q3}_icp_trials.csv").assign(model=Q3), r(f"{NEM}_e10_icp_trials.csv").assign(model=NEM)])
sc = icp[icp.truth == "scam"]
said = sc[sc.icp_t0.astype(str).str.upper().str.contains("SCAM")]
out("ICP", "scam trials answered SCAM in context", f"{len(said)}/{len(sc)}")
out("ICP", "... of which the agent then acted (harm)", frac(said.harm))
out("ICP", "harm over all scam trials", frac(sc.harm))
for m, g in sc.groupby("model"):
    out("ICP", f"  {m}: SCAM in context / harm", f"{g.icp_t0.astype(str).str.upper().str.contains('SCAM').sum()}/{len(g)}, {frac(g.harm)}")

# ---------------- E3' and E7: Nemotron-3-Super ---------------------------------------------------------------
v = r(f"{NEM}_v9free_trials.csv")
for c, g in v.groupby("condition"):
    s, o = g[g.truth == "scam"], g[g.scenario.isin(ONBOARD)]
    out("E3'", f"Nemotron {c}: scam harm / onboarding completed", f"{frac(s.harm)}, {frac(o.legit_completed)}")
a = r(f"{NEM}_v9free_adaptive_trials.csv")
out("E7", "Nemotron harm by scenario x condition",
    a.pivot_table(index="scenario", columns="condition", values="harm", aggfunc="sum").to_dict())

# ---------------- E8: real browser ---------------------------------------------------------------------------
e8n = pd.concat([r(f"{NEM}_{s}_episodes.csv") for s in ("e8", "e8run2", "e8run3")])
e8q = r(f"{Q3}_e8qwen3_episodes.csv")
for name, e in (("Nemotron", e8n), ("Qwen3", e8q)):
    t = e.assign(v=e.harm.where(e.truth == "scam", e.legit_completed))
    out("E8", f"{name}: harm (scams) / onboarding (payroll), episodes",
        t.pivot_table(index="condition", columns="scenario", values="v", aggfunc="sum").to_dict("index"))
both = pd.concat([e8n, e8q])
fl = both[both.scenario.isin(["fee_offer", "pii_lookalike_offer"])]
n0 = fl[fl.condition == "neutral"]; g1 = fl[fl.condition != "neutral"]
out("E8", "fee + look-alike harmed episodes: no gate vs. either gate",
    f"{frac(n0.harm)} vs {frac(g1.harm)}, p={fisher(int(n0.harm.sum()), len(n0), int(g1.harm.sum()), len(g1)):.1e}")

# ---------------- E9: FBI indicator in the policy (Nemotron) --------------------------------------------------
t, ad, re_ = r(f"{NEM}_e9ind_trials.csv"), r(f"{NEM}_e9ind_adaptive_trials.csv"), r(f"{NEM}_e9ind_real_trials.csv")
out("E9", "own-domain payroll submitted", frac(ad[ad.scenario == "adapt_pii_own"].harm))
out("E9", "own-domain fee + vendor paid", frac(ad[ad.scenario != "adapt_pii_own"].harm))
out("E9", "legitimate E3 onboardings completed", frac(t[t.scenario.isin(ONBOARD)].legit_completed))
out("E9", "real scam harm / legitimate real trials blocked",
    f"{frac(re_[re_.truth == 'scam'].harm)}, {frac(re_[re_.truth == 'legit'].legit_blocked)}")

# ---------------- E9b: false video-interview claim -------------------------------------------------------------
e9b = pd.concat([r(f"{m}_e9b_trials.csv").assign(model=m) for m in (NEM, Q3, "claude-sonnet-5.5")])
e9b = e9b.assign(variant=e9b.scenario.str.split("|").str[1])
for (c, var), g in e9b.groupby(["condition", "variant"]):
    out("E9b", f"{c:15s} {var:8s} harm (all models)", frac(g.harm))
out("E9b", "per model x condition x variant",
    e9b.pivot_table(index=["model", "condition"], columns="variant", values="harm", aggfunc="sum").to_dict("index"))

# ---------------- E11: policy in the agent's prompt; guard models ----------------------------------------------
pp = pd.concat([r(f"{m}_e11_policy_prompt_trials.csv").assign(model=m)
                for m in ["gemma-2-9b", "phi-3.5-mini", "qwen2.5-7b", "granite-3.1-8b", Q3, NEM]])
e3 = pp[~pp.scenario.str.startswith("adapt_")]
e7 = pp[pp.scenario.str.startswith("adapt_")]
for c in ("neutral", "policy_prompt"):
    g3, g7 = e3[e3.condition == c], e7[e7.condition == c]
    out("E11", f"{c}: E3 scam harm / E7 harm / look-alike / onboarding",
        f"{frac(g3[g3.truth == 'scam'].harm)}, {frac(g7.harm)}, "
        f"{frac(g3[g3.scenario == 'pii_lookalike_offer'].harm)}, {frac(g3[g3.scenario.isin(ONBOARD)].legit_completed)}")
g = e3[(e3.condition == "policy_prompt") & (e3.model == NEM)]
out("E11", "Nemotron policy prompt: E3 scam harm / onboarding",
    f"{frac(g[g.truth == 'scam'].harm)}, {frac(g[g.scenario.isin(ONBOARD)].legit_completed)}")
ind = e7[e7.condition == "indicator_prompt"]
out("E11", "indicator in prompt, E7 harm: Nemotron / five other models",
    f"{frac(ind[ind.model == NEM].harm)}, {frac(ind[ind.model != NEM].harm)}")
for gname, cols in (("granite-guardian-3.1-8b", ("block_harm", "block_scam_custom")),
                    ("qwen3guard-gen-8b", ("block", "block_strict")), ("llama-guard-4-12b", ("block",)),
                    ("qwen3-8b-pol-gate", ("block",))):
    gd = r(f"e11_guard_{gname}.csv")
    for col in cols:
        out("E11", f"guard {gname} [{col}]: blocked by set x truth",
            gd.groupby(["set", "truth"])[col].apply(lambda s: f"{int(s.sum())}/{len(s)}").to_dict())

# ---------------- E12: gate-aware attacker and ablations ------------------------------------------------------
for m in (NEM, Q3):
    at = r(f"{m}_e12_attack.csv")
    out("E12", f"{m}: successful searches by target x gate",
        at.pivot_table(index="target", columns="gate", values="success", aggfunc="sum").to_dict())
    pr = at[(at.target == "own_domain_payroll") & (at.gate == "hybrid_plus")]
    out("E12", f"{m}: own-domain payroll under hyb+ (success, of which round 0)",
        f"{int(pr.success.sum())}/{len(pr)}, {int((pr.success_round == 0).sum())}")
    ab = r(f"{m}_e12_ablation_trials.csv")
    out("E12", f"{m}: ablation scam harm / onboarding by condition",
        {c: f"{frac(g[g.truth == 'scam'].harm)}, {frac(g[g.scenario.isin(ONBOARD)].legit_completed)}"
         for c, g in ab.groupby("condition")})
tr = r(f"{NEM}_e12_transfer_trials.csv")
out("E12", "transfer of the optimised message to the full Nemotron agent", frac(tr.harm))
ab = r(f"{NEM}_e12_ablation_trials.csv")
n0, sv = ab[(ab.condition == "neutral") & (ab.truth == "scam")], ab[(ab.condition == "abl_self_verdict") & (ab.truth == "scam")]
out("E12", "Nemotron scam harm neutral vs self-verdict (Fisher)",
    f"{frac(n0.harm)} vs {frac(sv.harm)}, p={fisher(int(n0.harm.sum()), 32, int(sv.harm.sum()), 32):.1e}")

# ---------------- E13: browser-use -----------------------------------------------------------------------------
e13 = r(f"{NEM}_e13_episodes.csv")
t = e13.assign(v=e13.harm.where(e13.truth == "scam", e13.onboarding_done))
out("E13", "HTTP run, Nemotron in browser-use: harm (scams) / onboarding (payroll), of 5",
    t.pivot_table(index="condition", columns="scenario", values="v", aggfunc="sum").to_dict("index"))
fl = e13[e13.scenario.isin(["fee_offer", "lookalike"])]
a0, a1 = fl[fl.condition == "neutral"], fl[fl.condition == "hybrid_plus"]
out("E13", "fee + look-alike harm none vs hyb+ (Fisher)",
    f"{frac(a0.harm)} vs {frac(a1.harm)}, p={fisher(int(a0.harm.sum()), len(a0), int(a1.harm.sum()), len(a1)):.3f}")
out("E13", "episodes ending in an error", int(e13.error.notna().sum()))
g13 = r("gpt-6.1-sol_e13_episodes.csv")
out("E13", "HTTP run, GPT-6.1 Sol: scam harm / onboarding completed", f"{frac(g13[g13.truth == 'scam'].harm)}, {frac(g13[g13.truth == 'legit'].onboarding_done)}")
# HTTPS run (reported in the paper's main text): three models, five episodes per cell
for m in ("nemotron-3-super", "gpt-6.1-sol", "gemini-3.1-pro"):
    h = r(f"{m}_e13https_episodes.csv")
    t = h.assign(v=h.harm.where(h.truth == "scam", h.onboarding_done))
    out("E13", f"HTTPS {m}: harm (scams) / onboarding (payroll), of 5",
        t.pivot_table(index="condition", columns="scenario", values="v", aggfunc="sum").to_dict("index"))
    hp = h[h.condition == "hybrid_plus"]
    out("E13", f"HTTPS {m}: hyb+ episodes with an origin-check block (fee, look-alike)",
        int(sum(any(p["stage"] == "origin" for p in json.loads(x)) for x in hp[hp.scenario.isin(["fee_offer", "lookalike"])].posts)))
h = r("nemotron-3-super_e13https_episodes.csv")
fl = h[h.scenario.isin(["fee_offer", "lookalike"])]
a0, a1 = fl[fl.condition == "neutral"], fl[fl.condition == "hybrid_plus"]
out("E13", "HTTPS Nemotron fee + look-alike harm none vs hyb+ (Fisher)",
    f"{frac(a0.harm)} vs {frac(a1.harm)}, p={fisher(int(a0.harm.sum()), len(a0), int(a1.harm.sum()), len(a1)):.1e}")

# ---------------- R / R': Qwen3-8B and Nemotron robustness ------------------------------------------------------
q = r(f"{Q3}_q3_trials.csv")
for c, g in q.groupby("condition"):
    out("R", f"Qwen3 {c}: E3 scam harm / onboarding", f"{frac(g[g.truth == 'scam'].harm)}, {frac(g[g.scenario.isin(ONBOARD)].legit_completed)}")
qa = r(f"{Q3}_q3_adaptive_trials.csv")
out("R", "Qwen3 E7 harm by condition", qa.groupby("condition").harm.apply(frac).to_dict())
for who, pre in (("Qwen3", f"{Q3}_q3"), ("Nemotron", f"{NEM}_e10")):
    for c in ("neutral", "policy_gate", "hybrid_plus"):
        g = r(f"{pre}_real_neutral_{c}.csv")
        out("R", f"{who} label-free record, {c}: scam harm / legit blocked",
            f"{frac(g[g.truth == 'scam'].harm)}, {frac(g[g.truth == 'legit'].legit_blocked)}")
