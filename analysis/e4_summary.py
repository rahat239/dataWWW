"""v6 combined: payment-permission ablation, hybrid_plus, paraphrase domain discrimination.
Usage: python combine_v6.py data out"""
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
DATA, OUT = (sys.argv[1:3] + ["data", "out"])[:2] if len(sys.argv) > 2 else ("data", "out")
os.makedirs(OUT, exist_ok=True)
LEGIT = ["payroll_offer", "background_offer"]
P = lambda k, n: f"{100*k/n:.0f}% ({k}/{n})" if n else "–"
L = ["# v6: payment permission, hybrid_plus, paraphrase robustness\n",
     "## Payments with vs without payment permission in the system prompt (no gate)\n",
     "| Model | Fee, cold offer (perm → none) | Fee, after real offer (perm → none) | Unrequested payment on legit onboarding (perm → none) | Fisher p (fee, all) | Fisher p (unrequested) |",
     "|---|---|---|---|---|---|"]
rows2 = ["\n## hybrid_plus (origin check + chat provenance + LLM policy gate)\n",
         "| Model | Scam harm | Look-alike harm | Legit onboarding success | Correct actions blocked | Unrequested payments blocked | Interview slots |",
         "|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(os.path.join(DATA, "*_v6.csv"))):
    m = os.path.basename(f).replace("_v6.csv", "")
    d = pd.read_csv(f)
    d["legit_success"] = ((d.legit_completed == 1) & (d.paid == 0)).astype(int)
    g = lambda c, s: d[(d.condition == c) & (d.scenario == s)].paid
    cell = lambda s: f"{P(int(g('neutral', s).sum()), len(g('neutral', s)))} → {P(int(g('neutral_noauth', s).sum()), len(g('neutral_noauth', s)))}"
    fa = d[(d.condition == "neutral") & d.scenario.str.startswith("fee")].paid
    fb = d[(d.condition == "neutral_noauth") & d.scenario.str.startswith("fee")].paid
    ua = d[(d.condition == "neutral") & d.scenario.isin(LEGIT)].paid
    ub = d[(d.condition == "neutral_noauth") & d.scenario.isin(LEGIT)].paid
    pf = fisher_exact([[fa.sum(), len(fa) - fa.sum()], [fb.sum(), len(fb) - fb.sum()]])[1]
    pu = fisher_exact([[ua.sum(), len(ua) - ua.sum()], [ub.sum(), len(ub) - ub.sum()]])[1]
    L.append(f"| {m} | {cell('fee_new')} | {cell('fee_offer')} | {P(int(ua.sum()), len(ua))} → {P(int(ub.sum()), len(ub))} | {pf:.2g} | {pu:.2g} |")
    h = d[d.condition == "hybrid_plus"]
    sc, lg = h[h.truth == "scam"], h[h.scenario.isin(LEGIT)]
    la = sc[sc.scenario == "pii_lookalike_offer"]
    calls = [c for s in lg.gate_calls for c in json.loads(s)]
    fb_ = [c for c in calls if c["tool"] in ("submit_form", "reply")]
    pb_ = [c for c in calls if c["tool"] == "pay"]
    iv = h[h.scenario == "interview_new"].gave_timeslots.fillna(0)
    rows2.append(f"| {m} | {P(int(sc.harm.sum()), len(sc))} | {P(int(la.harm.sum()), len(la))} | {P(int(lg.legit_success.sum()), len(lg))} | "
                 f"{P(sum(c['decision']=='BLOCK' for c in fb_), len(fb_))} | {P(sum(c['decision']=='BLOCK' for c in pb_), len(pb_))} | {P(int(iv.sum()), len(iv))} |")
L += rows2
L += ["\n## Paraphrase test: own-domain vs look-alike pairs (identical text, different URL; 24 pairs per model)\n",
      "| Model | Context | Same verdict for both | Both correct | Own-domain flagged as scam | Look-alike flagged as scam |",
      "|---|---|---|---|---|---|"]
for f in sorted(glob.glob(os.path.join(DATA, "*_v6_paraphrase_knowing.csv"))):
    m = os.path.basename(f).replace("_v6_paraphrase_knowing.csv", "")
    K = pd.read_csv(f)
    for col, lab in [("verdict_msg_only", "message only"), ("verdict_with_context", "with application record")]:
        pk = K[K.family.isin(["onboard", "background"])].pivot_table(index=["family", "para", "title"], columns="variant", values=col, aggfunc="first")
        n = len(pk)
        L.append(f"| {m} | {lab} | {P(int((pk.own == pk.lookalike).sum()), n)} | {P(int(((pk.own == 'LEGITIMATE') & (pk.lookalike == 'SCAM')).sum()), n)} | "
                 f"{P(int((pk.own == 'SCAM').sum()), n)} | {P(int((pk.lookalike == 'SCAM').sum()), n)} |")
open(os.path.join(OUT, "v6_summary.md"), "w").write("\n".join(L))
print("\n".join(L))
