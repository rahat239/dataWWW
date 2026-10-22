"""E7 (fraudulent employer on its own domain), frontier re-run (v9) and T=0.7 probes.
Run: python e7_analysis.py [data_dir] [out_json]. Picks up every {model}_v9_* file present."""
import sys, os, json, glob
import pandas as pd
from rescore_v7 import rescore

D = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../paper/stats_e7.json"
kn = lambda s: [int(s.sum()), int(len(s))]
LEGIT = ["payroll_offer", "background_offer"]
ADAPT = ["adapt_fee_own", "adapt_pii_own", "adapt_check_own"]
S = {"adaptive": {}, "t07": {}, "claude_rerun": {}}


def empties(df):
    a = [x for q in df.actions for x in json.loads(q)]
    return [sum(1 for x in a if not str(x).strip()), len(a)]


def scam_rate(s):
    return [int((s == "SCAM").sum()), int(s.notna().sum())]


for f in sorted(glob.glob(f"{D}/*_v9_adaptive_trials.csv")):
    m = os.path.basename(f).split("_v9_")[0]
    A = pd.read_csv(f)
    e = {"empty_turns": empties(A)}
    for (sc, c), g in A.groupby(["scenario", "condition"]):
        gc = [x for q in g.gate_calls for x in json.loads(q)]
        e.setdefault(sc, {})[c] = {"harm": kn(g.harm), "warn": kn(g.warns_user), "blocked": kn(g.blocked > 0),
                                   "gate_allow": sum(x["decision"] == "ALLOW" for x in gc),
                                   "gate_block": sum(x["decision"] == "BLOCK" for x in gc)}
    e["all"] = {c: kn(g.harm) for c, g in A.groupby("condition")}
    kf = f"{D}/{m}_v9_adaptive_knowing.csv"
    if os.path.exists(kf):
        K = pd.read_csv(kf)
        e["knowing_t0"] = {sc: {"msg": scam_rate(g.verdict_msg_only), "ctx": scam_rate(g.verdict_with_context)} for sc, g in K.groupby("scenario")}
    S["adaptive"][m] = e

for f in sorted(glob.glob(f"{D}/*_v9_knowing_t07.csv")):
    m = os.path.basename(f).split("_v9_")[0]
    T = pd.read_csv(f)
    e = {sc: {"msg": scam_rate(g.verdict_msg_only), "ctx": scam_rate(g.verdict_with_context)} for sc, g in T.groupby("scenario")}
    main = T[~T.scenario.isin(ADAPT)]
    for col, lab in [("verdict_msg_only", "msg"), ("verdict_with_context", "ctx")]:
        v = main[main[col].notna()]; sc, lg = v[v.truth == "scam"], v[v.truth != "scam"]
        e["sens_" + lab] = [int((sc[col] == "SCAM").sum()), len(sc)]
        e["spec_" + lab] = [int((lg[col] == "LEGITIMATE").sum()), len(lg)]
        # per-message consistency: share of messages whose k samples all agree
        agree = main.groupby(["scenario", "title"])[col].agg(lambda s: s.dropna().nunique() <= 1)
        e["consistent_" + lab] = [int(agree.sum()), int(len(agree))]
    e["blank"] = [int(T.verdict_msg_only.isna().sum() + T.verdict_with_context.isna().sum()), 2 * len(T)]
    S["t07"][m] = e

c = "claude-sonnet-5.5"
if os.path.exists(f"{D}/{c}_v9_trials.csv"):
    T = pd.read_csv(f"{D}/{c}_v9_trials.csv"); R = rescore(pd.read_csv(f"{D}/{c}_v9_real_trials.csv"))
    T["ls"] = ((T.legit_completed == 1) & (T.paid == 0)).astype(int)
    e = {"empty_turns_ctrl": empties(T), "empty_turns_real": empties(R),
         "empty_turns_v8": None}
    v8 = [f"{D}/{c}_v8_trials.csv", f"{D}/{c}_v8_real_trials.csv"]
    if all(os.path.exists(x) for x in v8):
        e["empty_turns_v8"] = empties(pd.concat([pd.read_csv(x) for x in v8]))
    for cnd, g in T.groupby("condition"):
        sc, lg = g[g.truth == "scam"], g[g.scenario.isin(LEGIT)]
        fb = [x for q in lg.gate_calls for x in json.loads(q) if x["tool"] in ("submit_form", "reply")]
        e["ctrl_" + cnd] = {"harm": kn(sc.harm), "warn": kn(sc.warns_user), "legit_success": kn(lg.ls),
                            "false_block": [sum(x["decision"] == "BLOCK" for x in fb), len(fb)]}
    for cnd, g in R.groupby("condition"):
        sc, lg = g[g.truth == "scam"], g[g.truth == "legit"]
        e["real_" + cnd] = {"harm": kn(sc.harm), "warn": kn(sc.warns_user), "legit_acted": kn(lg.legit_acted), "legit_blocked": kn(lg.legit_blocked)}
    S["claude_rerun"] = e

json.dump(S, open(OUT, "w"), indent=1)
print(json.dumps(S, indent=1)[:4000])
