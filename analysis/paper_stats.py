"""All numbers and figures used in the paper. Run: python paper_stats.py data ../paper/figs"""
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, FIG = sys.argv[1], sys.argv[2]
os.makedirs(FIG, exist_ok=True)
S = {}
M = ["gemma-2-9b", "phi-3.5-mini", "qwen2.5-7b", "qwen2.5-14b", "granite-3.1-8b"]
LAB = {"gemma-2-9b": "Gemma-2-9B", "phi-3.5-mini": "Phi-3.5-mini", "qwen2.5-7b": "Qwen2.5-7B", "qwen2.5-14b": "Qwen2.5-14B", "granite-3.1-8b": "Granite-3.1-8B"}
COL = {"gemma-2-9b": "#2a78d6", "phi-3.5-mini": "#eb6834", "qwen2.5-7b": "#1baf7a", "qwen2.5-14b": "#6250d6", "granite-3.1-8b": "#c03e77"}
HATCH = {"gemma-2-9b": "", "phi-3.5-mini": "////", "qwen2.5-7b": "....", "qwen2.5-14b": "xxxx", "granite-3.1-8b": "\\\\\\\\"}
LEGIT = ["payroll_offer", "background_offer"]
import matplotlib.font_manager as fm, glob as _g
for _f in _g.glob(os.path.expanduser("~/.fonts/Libertinus*.otf")): fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Libertinus Serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2, "hatch.linewidth": 0.4,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "pdf.fonttype": 42})


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))


def kn(s): return [int(s.sum()), int(len(s))]


def fisher(a, b): return float(fisher_exact([[a[0], a[1]-a[0]], [b[0], b[1]-b[0]]])[1])


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)


def bars(ax, groups, models, get, labels, ylabel=None, annotate=True, err=True, rot=False):
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        for j, g in enumerate(groups):
            v = get(m, g)
            x = j + (i - (len(models)-1)/2) * w
            if v is None or v[1] == 0:
                ax.text(x, 2, "n/a", ha="center", va="bottom", fontsize=5.5, rotation=90, color="#888"); continue
            y = 100*v[0]/v[1]; lo, hi = wilson(*v)
            ax.bar(x, max(y, 0.8), w*0.9, color=COL[m], hatch=HATCH[m], edgecolor="white", lw=0.3,
                   label=LAB[m] if j == 0 else None)
            if err: ax.errorbar(x, y, yerr=[[max(0, y-100*lo)], [max(0, 100*hi-y)]], color="#333", lw=0.5, capsize=1.2)
            if annotate:
                top = 100*hi if err else y
                ax.text(x, top+1.5, f"{y:.0f}", ha="center", va="bottom", fontsize=5.0 if rot else 5.8,
                        rotation=90 if rot else 0)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels); ax.set_ylim(0, 116)
    if ylabel: ax.set_ylabel(ylabel)
    style(ax)


# ---------------- E1 (v3.1)
E1 = {}
for m in M[:3]:
    d = pd.read_csv(f"{D}/{m}_controlled.csv")
    E1[m] = {c: kn(d[d.condition == c].leaked_any) for c in ["neutral", "recognize", "privacy"]}
    n = d[d.condition == "neutral"]
    E1[m]["neutral_chat"] = kn(n[n.channel == "chat"].leaked_any)
    E1[m]["neutral_portal"] = kn(n[n.channel == "portal"].leaked_any)
    E1[m]["posting_scam"] = kn(n[n.fraudulent == 1].leaked_any); E1[m]["posting_legit"] = kn(n[n.fraudulent == 0].leaked_any)
    E1[m]["p_posting"] = fisher(E1[m]["posting_scam"], E1[m]["posting_legit"])
    E1[m]["p_recognize"] = fisher(E1[m]["recognize"], E1[m]["neutral"]); E1[m]["p_privacy"] = fisher(E1[m]["privacy"], E1[m]["neutral"])
    E1[m]["p_channel"] = fisher(E1[m]["neutral_portal"], E1[m]["neutral_chat"])
    E1[m]["n_trials"] = len(d)
    # reviewer fix: clear scams = no interview; legit-like = interview done, no urgency, portal (legitimate under our threat model)
    for c in ["neutral", "recognize", "privacy"]:
        g = d[d.condition == c]
        E1[m][c + "_scam"] = kn(g[g.interview == 0].leaked_any)
        E1[m][c + "_legitlike"] = kn(g[(g.interview == 1) & (g.urgency == 0) & (g.channel == "portal")].leaked_any)
    ns = n[n.interview == 0]
    E1[m]["scam_chat"] = kn(ns[ns.channel == "chat"].leaked_any); E1[m]["scam_portal"] = kn(ns[ns.channel == "portal"].leaked_any)
    E1[m]["p_scam_channel"] = fisher(E1[m]["scam_portal"], E1[m]["scam_chat"])
    E1[m]["p_scam_recognize"] = fisher(E1[m]["recognize_scam"], E1[m]["neutral_scam"])
    E1[m]["scam_posting_fraud"] = kn(ns[ns.fraudulent == 1].leaked_any); E1[m]["scam_posting_legit"] = kn(ns[ns.fraudulent == 0].leaked_any)
    E1[m]["p_scam_posting"] = fisher(E1[m]["scam_posting_fraud"], E1[m]["scam_posting_legit"])
S["E1"] = E1

# ---------------- E2 (v4)
RT = ["pii_chat", "pii_portal_company", "pii_portal_lookalike", "fee"]
E2 = {}
for m in M[:3]:
    f = f"{D}/{m}_v4.csv"
    E2[m] = {}
    if os.path.exists(f):
        d = pd.read_csv(f)
        for c in ["neutral", "generic_safety", "self_check_gate"]:
            g = d[(d.condition == c) & (d.rtype != "interview_only")]
            b = d[(d.condition == c) & (d.rtype == "interview_only")]
            E2[m][c] = {"harm": kn(g.harm), "attempted": kn(g.attempted_harm), "warn": kn(g.warns_user),
                        "slots": kn(b.gave_timeslots.fillna(0))} | {t: kn(g[g.rtype == t].harm) for t in RT}
    else:
        R = json.load(open(f"{D}/{m}_v4_summary.json"))["results"]
        k = lambda e: [int(round(e["rate"]*e["n"])), e["n"]]
        for c in ["neutral", "generic_safety", "self_check_gate"]:
            s = R[c]
            E2[m][c] = {"harm": k(s["harm_overall"]), "attempted": k(s["attempted_harm_overall"]), "warn": k(s["warns_user"]),
                        "slots": k(s["benign_control_timeslots"])} | {t: k(s["harm_by_request"][t]) for t in RT}
    for c in ["neutral", "generic_safety", "self_check_gate"]:
        if os.path.exists(f):
            g = d[(d.condition == c) & (d.rtype != "interview_only")]
            cell = lambda t, q: kn(g[(g.rtype == t) & (g.cue == q)].harm)
        else:
            h = R[c]["harm_by_request_and_cue"]; cell = lambda t, q, h=h: [int(round(h[f"{t}|{q}"]["rate"]*h[f"{t}|{q}"]["n"])), h[f"{t}|{q}"]["n"]]
        cl = {f"{t}|{q}": cell(t, q) for t in RT for q in ["scamlike", "legitlike"]}
        E2[m][c]["cells"] = cl
        sc = [k for k in cl if k not in ("pii_portal_company|legitlike", "pii_chat|legitlike")]
        E2[m][c]["clear_scam"] = [sum(cl[k][0] for k in sc), sum(cl[k][1] for k in sc)]
        E2[m][c]["clear_scam_data"] = [sum(cl[k][0] for k in sc if not k.startswith("fee")), sum(cl[k][1] for k in sc if not k.startswith("fee"))]
        E2[m][c]["legit_own"] = cl["pii_portal_company|legitlike"]
    E2[m]["p_fee_vs_data_gs_clear"] = fisher(E2[m]["generic_safety"]["fee"], E2[m]["generic_safety"]["clear_scam_data"])
    gs = E2[m]["generic_safety"]
    pii = [sum(gs[t][0] for t in RT[:3]), sum(gs[t][1] for t in RT[:3])]
    E2[m]["gs_pii"] = pii; E2[m]["p_fee_vs_pii_gs"] = fisher(gs["fee"], pii)
    E2[m]["p_gs"] = fisher(gs["harm"], E2[m]["neutral"]["harm"])
S["E2"] = E2

# ---------------- E3 (v5) + hybrid_plus (v6) + 14B (v6big)
def load_v5(m, runs=("v5", "v5_hybrid", "v6", "v6big", "v7_trials")):
    """All controlled runs of a model pooled (independent trial seeds; v7 repeats the main conditions)."""
    fr = []
    for r in runs:
        f = f"{D}/{m}_{r}.csv"
        if os.path.exists(f):
            d = pd.read_csv(f); d["run"] = r; fr.append(d)
    return pd.concat(fr, ignore_index=True)

G = ["neutral", "msg_gate", "ctx_gate", "policy_gate", "hybrid_gate", "hybrid_plus"]
E3 = {}
for m in M:
    d = load_v5(m); d["legit_success"] = ((d.legit_completed == 1) & (d.paid == 0)).astype(int)
    E3[m] = {}
    for c in G + ["neutral_noauth"]:
        g = d[d.condition == c]
        if g.empty: continue
        sc, lg = g[g.truth == "scam"], g[g.scenario.isin(LEGIT)]
        calls = [x for s in lg.gate_calls for x in json.loads(s)]
        fb = [x for x in calls if x["tool"] in ("submit_form", "reply")]
        pb = [x for x in calls if x["tool"] == "pay"]
        stg = [x.get("stage") for s in g.gate_calls for x in json.loads(s)]
        E3[m][c] = {"scam_harm": kn(sc.harm), "lookalike": kn(sc[sc.scenario == "pii_lookalike_offer"].harm),
                    "fee": kn(sc[sc.scenario.str.startswith("fee")].harm),
                    "fee_new": kn(sc[sc.scenario == "fee_new"].paid), "fee_offer": kn(sc[sc.scenario == "fee_offer"].paid),
                    "chat": kn(sc[sc.scenario == "pii_chat_new"].harm),
                    "attempted": kn(sc.attempted), "legit_success": kn(lg.legit_success),
                    "unreq_pay": kn(lg.paid), "false_block": [sum(x["decision"] == "BLOCK" for x in fb), len(fb)],
                    "pay_blocked": [sum(x["decision"] == "BLOCK" for x in pb), len(pb)],
                    "slots": kn(g[g.scenario == "interview_new"].gave_timeslots.fillna(0)),
                    "warn_scam": kn(sc.warns_user), "provenance_blocks": stg.count("provenance"),
                    "origin_blocks": stg.count("origin"), "n": len(g)}
S["E3"] = E3

# payment permission ablation (v6 for small models, v6big for 14B: same-run comparison)
PA = {}
for m in M:
    d = pd.read_csv(f"{D}/qwen2.5-14b_v6big.csv") if m == "qwen2.5-14b" else \
        pd.read_csv(f"{D}/granite-3.1-8b_v7_trials.csv") if m.startswith("granite") else pd.read_csv(f"{D}/{m}_v6.csv")
    r = {}
    for c in ["neutral", "neutral_noauth"]:
        g = d[d.condition == c]
        r[c] = {"fee_new": kn(g[g.scenario == "fee_new"].paid), "fee_offer": kn(g[g.scenario == "fee_offer"].paid),
                "fee": kn(g[g.scenario.str.startswith("fee")].paid), "unreq": kn(g[g.scenario.isin(LEGIT)].paid),
                "scam_harm": kn(g[g.truth == "scam"].harm)}
    r["p_fee"] = fisher(r["neutral"]["fee"], r["neutral_noauth"]["fee"])
    r["p_unreq"] = fisher(r["neutral"]["unreq"], r["neutral_noauth"]["unreq"])
    PA[m] = r
S["PA"] = PA

# knowing (standalone) for v5 scenarios
KN = {}
for m in M:
    f = {"qwen2.5-14b": f"{D}/qwen2.5-14b_knowing_v6big.csv", "granite-3.1-8b": f"{D}/granite-3.1-8b_v7_knowing.csv"}.get(m, f"{D}/{m}_knowing_v5.csv")
    k = pd.read_csv(f)
    KN[m] = {s: {"msg": kn(g.verdict_msg_only == "SCAM"), "ctx": kn(g.verdict_with_context == "SCAM")} for s, g in k.groupby("scenario")}
    sc = k[k.truth == "scam"]
    KN[m]["scam_msg"] = kn(sc.verdict_msg_only == "SCAM")
    KN[m]["fee_msg"] = kn(k[k.scenario.str.startswith("fee")].verdict_msg_only == "SCAM")
    lg = k[k.truth != "scam"]; KN[m]["legit_msg_ok"] = kn(lg.verdict_msg_only == "LEGITIMATE")
S["KN"] = KN

# paraphrase domain discrimination
PP = {}
for m in M:
    K = pd.read_csv(f"{D}/{m}_v7_paraphrase_knowing.csv" if m.startswith("granite") else f"{D}/{m}_v6_paraphrase_knowing.csv")
    PP[m] = {}
    for col, lab in [("verdict_msg_only", "msg"), ("verdict_with_context", "ctx")]:
        pk = K[K.family.isin(["onboard", "background"])].pivot_table(index=["family", "para", "title"], columns="variant", values=col, aggfunc="first")
        n = len(pk)
        PP[m][lab] = {"same": [int((pk.own == pk.lookalike).sum()), n], "both": [int(((pk.own == "LEGITIMATE") & (pk.lookalike == "SCAM")).sum()), n],
                      "own_flag": [int((pk.own == "SCAM").sum()), n], "look_flag": [int((pk.lookalike == "SCAM").sum()), n]}
        fee = K[K.family == "fee"]; PP[m][lab]["fee_flag"] = kn(fee[col] == "SCAM")
        ch = K[K.family == "pii_chat"]; PP[m][lab]["chat_flag"] = kn(ch[col] == "SCAM")
S["PP"] = PP

# ---------------- E5: real messages (v7). Off-platform harm = message to a scammer-supplied contact (rescore_v7).
from rescore_v7 import rescore
E5 = {}
ASKS = ["personal_data", "contact_offplatform", "deposit_check", "click_link_or_form", "pay_fee", "interview_only", "other"]
for m in M:
    r = pd.read_csv(f"{D}/{m}_v7_real_trials.csv")
    fx = f"{D}/{m}_v7fix_real_trials.csv"   # Gemma hybrid_plus re-run after the own-contact origin fix
    if os.path.exists(fx):
        f = pd.read_csv(fx); r = pd.concat([r[~r.condition.isin(f.condition.unique())], f], ignore_index=True)
    r = rescore(r); k = pd.read_csv(f"{D}/{m}_v7_real_knowing.csv")
    knew = k.set_index("id").verdict_msg_only.eq("SCAM")
    e = {"know_scam": kn(k[k.label == "scam"].verdict_msg_only == "SCAM"),
         "know_scam_ctx": kn(k[k.label == "scam"].verdict_with_context == "SCAM"),
         "flag_legit": kn(k[k.label == "legit"].verdict_msg_only == "SCAM"),
         "flag_legit_ctx": kn(k[k.label == "legit"].verdict_with_context == "SCAM")}
    for c in ["neutral", "policy_gate", "hybrid_plus"]:
        g = r[r.condition == c]; sc = g[g.truth == "scam"]; lg = g[g.truth == "legit"]
        e[c] = {"harm": kn(sc.harm), "data": kn(sc.harm_data), "money": kn(sc.harm_money), "off": kn(sc.harm_offplatform),
                "warn": kn(sc.warns_user), "attempted": kn(sc.attempted), "legit_acted": kn(lg.legit_acted),
                "legit_blocked": kn(lg.legit_blocked), "msgs_harmed": [int(sc.groupby("msg_id").harm.max().sum()), int(sc.msg_id.nunique())]}
        e[c]["by_ask"] = {a: kn(sc[sc.ask == a].harm) for a in ASKS}
        if c != "neutral": e[c]["p"] = fisher(e[c]["harm"], e["neutral"]["harm"])
    n = r[(r.condition == "neutral") & (r.truth == "scam")]
    e["harm_when_known"] = kn(n[n.msg_id.map(knew)].harm); e["harm_when_unknown"] = kn(n[~n.msg_id.map(knew)].harm)
    e["n_trials"] = len(r)
    E5[m] = e
tot = lambda c, key: [sum(E5[m][c][key][0] for m in M), sum(E5[m][c][key][1] for m in M)]
E5["all"] = {c: {key: tot(c, key) for key in ["harm", "data", "money", "off", "legit_acted", "legit_blocked", "warn"]}
             for c in ["neutral", "policy_gate", "hybrid_plus"]}
E5["all"]["by_ask"] = {a: [sum(E5[m]["neutral"]["by_ask"][a][0] for m in M), sum(E5[m]["neutral"]["by_ask"][a][1] for m in M)] for a in ASKS}
E5["all"]["harm_when_known"] = [sum(E5[m]["harm_when_known"][i] for m in M) for i in (0, 1)]
S["E5"] = E5

# ---------------- E6: frontier API models (OpenRouter). Claude's probes come from a re-run (v8b) after empty-answer retries.
FM = {"claude-sonnet-5.5": "Claude Sonnet 5.5", "gpt-6.1-sol": "GPT-6.1 Sol", "gemini-3.1-pro": "Gemini 3.1 Pro"}
E6 = {}
def valid_kn(s): return [int((s == "SCAM").sum()), int(s.notna().sum())]
for m in FM:
    kv = "v8b" if m.startswith("claude") else "v8"
    K = pd.read_csv(f"{D}/{m}_{kv}_knowing.csv"); P = pd.read_csv(f"{D}/{m}_{kv}_paraphrase_knowing.csv")
    R = pd.read_csv(f"{D}/{m}_{kv}_real_knowing.csv")
    e = {"know_scam": valid_kn(K[K.truth == "scam"].verdict_msg_only),
         "know_legitpii": valid_kn(K[K.scenario.isin(LEGIT)].verdict_msg_only),
         "real_scam": valid_kn(R[R.label == "scam"].verdict_msg_only), "real_legit": valid_kn(R[R.label == "legit"].verdict_msg_only),
         "probe_blank": [int(K.verdict_msg_only.isna().sum() + K.verdict_with_context.isna().sum() + R.verdict_msg_only.isna().sum() + R.verdict_with_context.isna().sum()), 2 * (len(K) + len(R))]}
    x = P[P.family.isin(["onboard", "background"])]
    for col, lab in [("verdict_msg_only", "msg"), ("verdict_with_context", "ctx")]:
        pk = x.pivot_table(index=["family", "para", "title"], columns="variant", values=col, aggfunc="first").dropna()
        e["pairs_" + lab] = [int(((pk.own == "LEGITIMATE") & (pk.lookalike == "SCAM")).sum()), int(len(pk))]
    T = pd.read_csv(f"{D}/{m}_v8_trials.csv"); T["legit_success"] = ((T.legit_completed == 1) & (T.paid == 0)).astype(int)
    for c, g in T.groupby("condition"):
        sc, lg = g[g.truth == "scam"], g[g.scenario.isin(LEGIT)]
        fb = [x for q in lg.gate_calls for x in json.loads(q) if x["tool"] in ("submit_form", "reply")]
        e["ctrl_" + c] = {"harm": kn(sc.harm), "lookalike": kn(sc[sc.scenario == "pii_lookalike_offer"].harm),
                          "fee": kn(sc[sc.scenario.str.startswith("fee")].harm), "warn": kn(sc.warns_user),
                          "legit_success": kn(lg.legit_success), "false_block": [sum(x["decision"] == "BLOCK" for x in fb), len(fb)]}
    r = rescore(pd.read_csv(f"{D}/{m}_v8_real_trials.csv"))
    for c, g in r.groupby("condition"):
        sc, lg = g[g.truth == "scam"], g[g.truth == "legit"]
        e["real_" + c] = {"harm": kn(sc.harm), "warn": kn(sc.warns_user), "legit_acted": kn(lg.legit_acted), "legit_blocked": kn(lg.legit_blocked)}
    acts = [a for q in pd.concat([T.actions, r.actions]) for a in json.loads(q)]
    e["empty_turns"] = [sum(1 for a in acts if not str(a).strip()), len(acts)]
    e["finished"] = [int(T.finished.sum() + r.finished.sum()), len(T) + len(r)]
    e["n_trials"] = len(T) + len(r)
    E6[m] = e
S["E6"] = E6

# totals
ctrl = {m: load_v5(m) for m in M}
S["trials"] = {"E1": sum(E1[m]["n_trials"] for m in E1), "E2": 720,
               "E3_E4": sum(len(ctrl[m]) for m in M), "E5": sum(E5[m]["n_trials"] for m in M), "E6": sum(E6[m]["n_trials"] for m in E6)}
S["trials"]["total"] = sum(S["trials"].values())
json.dump(S, open(f"{FIG}/../stats.json", "w"), indent=1)

# ================= FIGURES
def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{FIG}/{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)

def top_legend(ax, ncol, handles=None):
    h, l = ax.get_legend_handles_labels() if handles is None else handles
    ax.legend(h, l, frameon=False, ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=5.8,
              handlelength=1.6, columnspacing=1.0)

def ylim130(ax):
    ax.set_ylim(0, 132); ax.set_yticks([0, 20, 40, 60, 80, 100])

RTL = ["PII via\nchat", "PII via own-\ndomain portal", "PII via look-\nalike portal", "$89\nfee"]
SHORT = {"gemma-2-9b": "Gemma-2\n9B", "phi-3.5-mini": "Phi-3.5\nmini", "qwen2.5-7b": "Qwen2.5\n7B", "qwen2.5-14b": "Qwen2.5\n14B", "granite-3.1-8b": "Granite-3.1\n8B"}

# Fig 2
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [1.0, 1.45], "wspace": 0.2})
ax = axes[0]; w = 0.27
for i, m in enumerate(M):
    for off, v, face, hatch in [(-w, KN[m]["fee_msg"], "#c3c2b7", ""), (0, KN[m]["legit_msg_ok"], "#e8e7e1", "...."),
                                (w, E3[m]["neutral"]["fee"], COL[m], HATCH[m])]:
        y = 100*v[0]/v[1]; lo, hi = wilson(*v)
        ax.bar(i+off, max(y, 0.8), w*0.92, color=face, edgecolor="#9a998f" if face == "#e8e7e1" else "white", lw=0.3, hatch=hatch)
        ax.errorbar(i+off, y, yerr=[[max(0, y-100*lo)], [max(0, 100*hi-y)]], color="#333", lw=0.5, capsize=1.0)
        ax.text(i+off, 100*hi+1.5, f"{y:.0f}", ha="center", va="bottom", fontsize=5.4, rotation=90)
from matplotlib.patches import Patch
top_legend(ax, 1, ([Patch(color="#c3c2b7"), Patch(facecolor="#e8e7e1", edgecolor="#9a998f", hatch="...."), Patch(facecolor="#6b6a66")],
                   ["Fee requests labelled SCAM (standalone)", "Legitimate requests labelled LEGITIMATE", "Fee paid when acting as agent (colored)"]))
ax.set_xticks(range(len(M))); ax.set_xticklabels([SHORT[m] for m in M], fontsize=6.6); ax.set_ylim(0, 150); ax.set_yticks([0, 20, 40, 60, 80, 100]); style(ax)
ax.set_ylabel("Messages / trials (%)"); ax.set_title("(a) Fee: knowing vs. doing, no defense (E3/E4)")
ax = axes[1]
RT2 = ["pii_chat|scamlike", "pii_portal_company|scamlike", "pii_portal_lookalike|*", "fee|*", "pii_portal_company|legitlike"]
def e2cell(m, key):
    t, q = key.split("|"); cl = E2[m]["neutral"]["cells"]
    if q != "*": return cl[key]
    a, b = cl[f"{t}|scamlike"], cl[f"{t}|legitlike"]; return [a[0]+b[0], a[1]+b[1]]
bars(ax, RT2, M[:3], e2cell, ["PII by chat,\nno interview", "PII own domain,\nno interview", "PII look-\nalike portal", "\\$89\nfee", "Own domain after\ninterview (legit.)"])
ax.tick_params(axis="x", labelsize=6.6)
ax.axvline(3.5, color="#9a998f", lw=0.6, ls="--"); ylim130(ax); top_legend(ax, 3)
ax.set_title("(b) Data sent / fee paid by request, no defense (E2)")
save(fig, "fig_knowing_doing")

# Fig 3
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0), gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.12})
bars(axes[0], ["neutral", "recognize", "privacy"], M[:3], lambda m, c: E1[m][c + "_scam"],
     ["No\ninstruction", "\"Assess scam\nrisk first\"", "\"Never share\nID/DOB/bank\""], ylabel="Trials with harm (%)")
ylim130(axes[0]); top_legend(axes[0], 3); axes[0].set_title("(a) User instructions (E1, no-interview scams)")
bars(axes[1], RT2[:4], M[:3], lambda m, k: (lambda t, q: E2[m]["generic_safety"]["cells"][k] if q != "*" else
     [E2[m]["generic_safety"]["cells"][f"{t}|scamlike"][0] + E2[m]["generic_safety"]["cells"][f"{t}|legitlike"][0], 16])(*k.split("|")),
     ["PII by chat,\nno interview", "PII own domain,\nno interview", "PII look-\nalike portal", "\\$89\nfee"])
axes[1].tick_params(axis="x", labelsize=6.6); ylim130(axes[1]); top_legend(axes[1], 3)
axes[1].set_title("(b) Generic safety line (E2, scam requests)")
save(fig, "fig_prompt_defenses")

# Fig 4
GL = ["None", "Msg", "Ctx", "Pol", "Hyb", "Hyb+"]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), sharey=True, gridspec_kw={"wspace": 0.06})
for ax, key, title in [(axes[0], "scam_harm", "(a) Scam trials harmed ↓"),
                       (axes[1], "lookalike", "(b) Look-alike portal harmed ↓"),
                       (axes[2], "legit_success", "(c) Real onboarding completed ↑")]:
    bars(ax, G, M, lambda m, c: E3[m].get(c, {}).get(key), GL, annotate=True, err=False, rot=True)
    ylim130(ax); ax.set_title(title)
axes[0].set_ylabel("Trials (%)")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.95), fontsize=7.2)
for ax in axes: ax.set_ylim(0, 112)
save(fig, "fig_gates")

# Fig 5
fig, ax = plt.subplots(figsize=(3.33, 2.5))
rows = [(m, c) for m in M for c in ["msg", "ctx"]]
for i, (m, c) in enumerate(rows):
    y = len(rows) - 1 - i
    o = 100*PP[m][c]["own_flag"][0]/PP[m][c]["own_flag"][1]; l = 100*PP[m][c]["look_flag"][0]/PP[m][c]["look_flag"][1]
    ax.plot([o, l], [y, y], color="#c3c2b7", lw=1.4, zorder=1)
    ax.scatter([o], [y], s=22, facecolor="white", edgecolor=COL[m], lw=1.2, zorder=3, marker="o")
    ax.scatter([l], [y], s=18, color=COL[m], zorder=2, marker="s")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{LAB[m]}, {'message only' if c=='msg' else '+ app. record'}" for m, c in rows][::-1], fontsize=7)
ax.set_xlim(-5, 105); ax.set_xlabel("Labelled SCAM (%), 24 identical-text pairs")
ax.set_ylim(-0.6, len(rows) + 0.9)
ax.legend([plt.Line2D([], [], ls="", marker="o", mfc="white", mec="#333"), plt.Line2D([], [], ls="", marker="s", color="#333")],
          ["Own-domain link (legitimate)", "Look-alike link (scam)"], frameon=False, fontsize=6.8, loc="upper center", ncol=2,
          bbox_to_anchor=(0.45, 1.0), handletextpad=0.3)
ax.spines[["top", "right"]].set_visible(False); ax.xaxis.grid(True, color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
save(fig, "fig_domain_blindness")
print(json.dumps(S["trials"]), "ok")

# Fig 6: E5 real messages — knowing vs doing and gates
fig, ax = plt.subplots(figsize=(3.33, 2.3))
series = [("know", "Labelled SCAM standalone", "#c3c2b7", ""), ("neutral", "Harm, no gate", "#d6453d", ""),
          ("policy_gate", "Harm, Pol gate", "#eb9a34", "////"), ("hybrid_plus", "Harm, Hyb+ gate", "#1baf7a", "")]
w = 0.2
for i, m in enumerate(M):
    for j, (key, lab, col, h) in enumerate(series):
        v = E5[m]["know_scam"] if key == "know" else E5[m][key]["harm"]
        y = 100*v[0]/v[1]; lo, hi = wilson(*v); x = i + (j-1.5)*w
        ax.bar(x, max(y, 0.8), w*0.9, color=col, hatch=h, edgecolor="white", lw=0.3, label=lab if i == 0 else None)
        ax.errorbar(x, y, yerr=[[max(0, y-100*lo)], [max(0, 100*hi-y)]], color="#333", lw=0.45, capsize=1)
        ax.text(x, 100*hi+1.5, f"{y:.0f}", ha="center", va="bottom", fontsize=5.2, rotation=90)
ax.set_xticks(range(len(M))); ax.set_xticklabels([SHORT[m] for m in M], fontsize=7)
ax.set_ylim(0, 150); ax.set_yticks([0, 20, 40, 60, 80, 100]); style(ax)
ax.set_ylabel("Messages / trials (%)")
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=6.8, handlelength=1.4, columnspacing=0.8)
save(fig, "fig_real")
print("fig_real ok")
