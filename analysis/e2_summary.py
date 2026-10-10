"""Combine v4 (request type x defense) results across models.

Usage: python combine_v4.py data/ out/
Reads <model>_v4.csv (trial level) when present, otherwise <model>_v4_summary.json.
Harmful-request trials only (interview_only is the benign control).
"""
import sys, os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(OUT, exist_ok=True)
COND = ["neutral", "generic_safety", "self_check_gate"]
CL = {"neutral": "Neutral", "generic_safety": "Generic safety prompt", "self_check_gate": "Self-check gate"}
RT = ["pii_chat", "pii_portal_company", "pii_portal_lookalike", "fee"]
RL = {"pii_chat": "PII in chat", "pii_portal_company": "PII, company portal",
      "pii_portal_lookalike": "PII, look-alike portal", "fee": "$89 fee"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


# counts[model][cond][metric] = (k, n); per-request harm in counts[model][cond]["harm|rt"]
counts = {}
models = sorted({os.path.basename(f).split("_v4")[0] for f in glob.glob(os.path.join(DATA, "*_v4*"))})
for m in models:
    csv = os.path.join(DATA, f"{m}_v4.csv")
    js = os.path.join(DATA, f"{m}_v4_summary.json")
    counts[m] = {}
    if os.path.exists(csv):
        d = pd.read_csv(csv)
        for c in COND:
            g = d[d.condition == c]
            h = g[g.rtype != "interview_only"]
            b = g[g.rtype == "interview_only"]
            r = {k: (int(h[k].sum()), len(h)) for k in ["harm", "attempted_harm", "warns_user"]}
            r["timeslots"] = (int(b.gave_timeslots.fillna(0).sum()), len(b))
            for t in RT:
                x = h[h.rtype == t]
                r["harm|" + t] = (int(x.harm.sum()), len(x))
            counts[m][c] = r
    else:
        R = json.load(open(js))["results"]
        for c in COND:
            s = R[c]
            k = lambda e: (int(round(e["rate"] * e["n"])), e["n"])
            r = {"harm": k(s["harm_overall"]), "attempted_harm": k(s["attempted_harm_overall"]),
                 "warns_user": k(s["warns_user"]), "timeslots": k(s["benign_control_timeslots"])}
            for t in RT:
                r["harm|" + t] = k(s["harm_by_request"][t])
            counts[m][c] = r

# ---------- table
rows = []
for m in models:
    for c in COND:
        r = counts[m][c]
        row = {"model": m, "condition": c}
        for key, (k, n) in r.items():
            row[key] = k / n if n else np.nan
            row[key + "_n"] = n
        rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "v4_table.csv"), index=False)

# ---------- tests vs neutral, and fee vs PII under generic safety
tests = []
for m in models:
    k0, n0 = counts[m]["neutral"]["harm"]
    for c in ["generic_safety", "self_check_gate"]:
        k1, n1 = counts[m][c]["harm"]
        p = fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])[1]
        tests.append({"model": m, "test": f"{c} vs neutral (harm)", "a": k1 / n1, "b": k0 / n0, "p": p})
    kf, nf = counts[m]["generic_safety"]["harm|fee"]
    kp = sum(counts[m]["generic_safety"]["harm|" + t][0] for t in RT[:3])
    np_ = sum(counts[m]["generic_safety"]["harm|" + t][1] for t in RT[:3])
    p = fisher_exact([[kf, nf - kf], [kp, np_ - kp]])[1]
    tests.append({"model": m, "test": "generic safety: fee vs PII (harm)", "a": kf / nf, "b": kp / np_, "p": p})
TS = pd.DataFrame(tests)
TS.to_csv(os.path.join(OUT, "v4_tests.csv"), index=False)

# ---------- figure: (a) harm by defense, (b) harm by request type under generic safety
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
colors = ["#2a6fdb", "#e0732f", "#3a9e5f", "#8e5cc8"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), dpi=200, gridspec_kw={"width_ratios": [3, 4]})


def bars(ax, groups, getter, labels):
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        ks = [getter(m, g) for g in groups]
        y = np.array([k / n * 100 for k, n in ks])
        ci = np.array([wilson(k, n) for k, n in ks]) * 100
        xs = np.arange(len(groups)) + (i - (len(models) - 1) / 2) * w
        ax.bar(xs, y, w * 0.92, yerr=[y - ci[:, 0], ci[:, 1] - y], capsize=1.5, color=colors[i],
               label=m, error_kw=dict(lw=0.7))
        for xx, yy, hi in zip(xs, y, ci[:, 1]):
            ax.text(xx, hi + 1.5, f"{yy:.0f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylim(0, 118)
    ax.tick_params(labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)


bars(axes[0], COND, lambda m, c: counts[m][c]["harm"], ["Neutral", "Generic\nsafety prompt", "Self-check\ngate"])
axes[0].set_ylabel("Harmful trials (%)", fontsize=7)
axes[0].set_title("(a) Harm by defense", fontsize=7.5)
bars(axes[1], RT, lambda m, t: counts[m]["generic_safety"]["harm|" + t],
     ["PII\nchat", "PII company\nportal", "PII look-alike\nportal", "$89\nfee"])
axes[1].set_title("(b) Generic safety prompt, by request type", fontsize=7.5)
axes[0].legend(fontsize=6, frameon=False, loc="upper right", ncol=1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_v4_defenses.png"))
fig.savefig(os.path.join(OUT, "fig_v4_defenses.pdf"))

# ---------- markdown
P = lambda kn: f"{100 * kn[0] / kn[1]:.0f}% ({kn[0]}/{kn[1]})"
L = ["# v4: request type x defense, combined\n",
     "| Model | Condition | Harm | Attempted | Warns user | Benign: offers timeslots |", "|---|---|---|---|---|---|"]
for m in models:
    for c in COND:
        r = counts[m][c]
        L.append(f"| {m} | {CL[c]} | {P(r['harm'])} | {P(r['attempted_harm'])} | {P(r['warns_user'])} | {P(r['timeslots'])} |")
L += ["\n## Harm by request type\n", "| Model | Condition | " + " | ".join(RL[t] for t in RT) + " |",
      "|---|---|" + "---|" * len(RT)]
for m in models:
    for c in COND:
        L.append(f"| {m} | {CL[c]} | " + " | ".join(P(counts[m][c]['harm|' + t]) for t in RT) + " |")
L += ["\n## Fisher exact tests\n", "| Model | Test | A | B | p |", "|---|---|---|---|---|"]
for _, r in TS.iterrows():
    L.append(f"| {r.model} | {r.test} | {100 * r.a:.0f}% | {100 * r.b:.0f}% | {r.p:.2g} |")
open(os.path.join(OUT, "v4_summary.md"), "w").write("\n".join(L))
print("\n".join(L))
