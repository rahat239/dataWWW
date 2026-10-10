"""All numbers and figures used in the paper.
Run from the repository root:  python analysis/paper_stats.py data results/figures
Writes results/stats.json and the four plotted figures."""
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, FIG = sys.argv[1], sys.argv[2]
os.makedirs(FIG, exist_ok=True)
S = {}
M = ["gemma-2-9b", "phi-3.5-mini", "qwen2.5-7b", "qwen2.5-14b"]
LAB = {"gemma-2-9b": "Gemma-2-9B", "phi-3.5-mini": "Phi-3.5-mini", "qwen2.5-7b": "Qwen2.5-7B", "qwen2.5-14b": "Qwen2.5-14B"}
COL = {"gemma-2-9b": "#2a78d6", "phi-3.5-mini": "#eb6834", "qwen2.5-7b": "#1baf7a", "qwen2.5-14b": "#6250d6"}
HATCH = {"gemma-2-9b": "", "phi-3.5-mini": "////", "qwen2.5-7b": "....", "qwen2.5-14b": "xxxx"}
LEGIT = ["payroll_offer", "background_offer"]
import matplotlib.font_manager as fm, glob as _g
for _f in _g.glob(os.path.expanduser("~/.fonts/Libertinus*.otf")): fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": ["Libertinus Serif", "DejaVu Serif"], "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "hatch.linewidth": 0.4,
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
                ax.text(x, 2, "n/a", ha="center", va="bottom", fontsize=4.5, rotation=90, color="#888"); continue
            y = 100*v[0]/v[1]; lo, hi = wilson(*v)
            ax.bar(x, max(y, 0.8), w*0.9, color=COL[m], hatch=HATCH[m], edgecolor="white", lw=0.3,
                   label=LAB[m] if j == 0 else None)
            if err: ax.errorbar(x, y, yerr=[[y-100*lo], [100*hi-y]], color="#333", lw=0.5, capsize=1.2)
            if annotate:
                top = 100*hi if err else y
                ax.text(x, top+1.5, f"{y:.0f}", ha="center", va="bottom", fontsize=4.4 if rot else 4.6,
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
    gs = E2[m]["generic_safety"]
    pii = [sum(gs[t][0] for t in RT[:3]), sum(gs[t][1] for t in RT[:3])]
    E2[m]["gs_pii"] = pii; E2[m]["p_fee_vs_pii_gs"] = fisher(gs["fee"], pii)
    E2[m]["p_gs"] = fisher(gs["harm"], E2[m]["neutral"]["harm"])
S["E2"] = E2

# ---------------- E3 (v5) + hybrid_plus (v6) + 14B (v6big)
def load_v5(m):
    if m == "qwen2.5-14b":
        return pd.read_csv(f"{D}/qwen2.5-14b_v6big.csv")
    d = pd.read_csv(f"{D}/{m}_v5.csv")
    h = f"{D}/{m}_v5_hybrid.csv"
    if os.path.exists(h): d = pd.concat([d, pd.read_csv(h)])
    v6 = pd.read_csv(f"{D}/{m}_v6.csv")
    return pd.concat([d, v6[v6.condition.isin(["hybrid_plus", "neutral_noauth"])]])

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
    d = pd.read_csv(f"{D}/qwen2.5-14b_v6big.csv") if m == "qwen2.5-14b" else pd.read_csv(f"{D}/{m}_v6.csv")
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
    f = f"{D}/{m}_knowing_v5.csv" if m != "qwen2.5-14b" else f"{D}/qwen2.5-14b_knowing_v6big.csv"
    k = pd.read_csv(f)
    KN[m] = {s: {"msg": kn(g.verdict_msg_only == "SCAM"), "ctx": kn(g.verdict_with_context == "SCAM")} for s, g in k.groupby("scenario")}
    sc = k[k.truth == "scam"]
    KN[m]["scam_msg"] = kn(sc.verdict_msg_only == "SCAM")
S["KN"] = KN

# paraphrase domain discrimination
PP = {}
for m in M:
    K = pd.read_csv(f"{D}/{m}_v6_paraphrase_knowing.csv")
    PP[m] = {}
    for col, lab in [("verdict_msg_only", "msg"), ("verdict_with_context", "ctx")]:
        pk = K[K.family.isin(["onboard", "background"])].pivot_table(index=["family", "para", "title"], columns="variant", values=col, aggfunc="first")
        n = len(pk)
        PP[m][lab] = {"same": [int((pk.own == pk.lookalike).sum()), n], "both": [int(((pk.own == "LEGITIMATE") & (pk.lookalike == "SCAM")).sum()), n],
                      "own_flag": [int((pk.own == "SCAM").sum()), n], "look_flag": [int((pk.lookalike == "SCAM").sum()), n]}
        fee = K[K.family == "fee"]; PP[m][lab]["fee_flag"] = kn(fee[col] == "SCAM")
        ch = K[K.family == "pii_chat"]; PP[m][lab]["chat_flag"] = kn(ch[col] == "SCAM")
S["PP"] = PP

# totals
S["trials"] = {"E1": sum(E1[m]["n_trials"] for m in E1), "E2": 720,
               "E3": sum(len(load_v5(m)[lambda x: ~x.condition.isin(["hybrid_plus", "neutral_noauth"])]) for m in M[:3]),
               "E4": sum(len(pd.read_csv(f"{D}/{m}_v6.csv")) for m in M[:3]) + len(pd.read_csv(f"{D}/qwen2.5-14b_v6big.csv"))}
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
SHORT = {"gemma-2-9b": "Gemma-2\n9B", "phi-3.5-mini": "Phi-3.5\nmini", "qwen2.5-7b": "Qwen2.5\n7B", "qwen2.5-14b": "Qwen2.5\n14B"}

# Fig 2
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0), gridspec_kw={"width_ratios": [1.0, 1.45], "wspace": 0.16})
ax = axes[0]; w = 0.36
for i, m in enumerate(M):
    for off, v, face, hatch in [(-w/2, KN[m]["scam_msg"], "#c3c2b7", ""), (w/2, E3[m]["neutral"]["scam_harm"], COL[m], HATCH[m])]:
        y = 100*v[0]/v[1]; lo, hi = wilson(*v)
        ax.bar(i+off, y, w*0.92, color=face, edgecolor="white", lw=0.3, hatch=hatch)
        ax.errorbar(i+off, y, yerr=[[y-100*lo], [100*hi-y]], color="#333", lw=0.5, capsize=1.2)
        ax.text(i+off, 100*hi+1.5, f"{y:.0f}", ha="center", va="bottom", fontsize=4.8)
from matplotlib.patches import Patch
top_legend(ax, 1, ([Patch(color="#c3c2b7"), Patch(facecolor="#6b6a66")],
                   ["Labelled SCAM when shown standalone", "Harmed when acting as agent (colored)"]))
ax.set_xticks(range(len(M))); ax.set_xticklabels([SHORT[m] for m in M]); ylim130(ax); style(ax)
ax.set_ylabel("Scam messages / trials (%)"); ax.set_title("(a) Knowing vs. doing, no defense (E3/E4)")
ax = axes[1]
bars(ax, RT, M[:3], lambda m, t: E2[m]["neutral"][t], RTL); ylim130(ax); top_legend(ax, 3)
ax.set_title("(b) Harm by request type, no defense (E2, n=16/bar)")
save(fig, "fig_knowing_doing")

# Fig 3
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0), gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.12})
bars(axes[0], ["neutral", "recognize", "privacy"], M[:3], lambda m, c: E1[m][c],
     ["No\ninstruction", "\"Assess scam\nrisk first\"", "\"Never share\nID/DOB/bank\""], ylabel="Trials with harm (%)")
ylim130(axes[0]); top_legend(axes[0], 3); axes[0].set_title("(a) User instructions (E1, n=96/bar)")
bars(axes[1], RT, M[:3], lambda m, t: E2[m]["generic_safety"][t], RTL); ylim130(axes[1]); top_legend(axes[1], 3)
axes[1].set_title("(b) Generic safety line in system prompt (E2, n=16/bar)")
save(fig, "fig_prompt_defenses")

# Fig 4
GL = ["None", "Msg", "Ctx", "Pol", "Hyb", "Hyb+"]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.95), sharey=True, gridspec_kw={"wspace": 0.06})
for ax, key, title in [(axes[0], "scam_harm", "(a) Scam trials harmed ↓"),
                       (axes[1], "lookalike", "(b) Look-alike portal harmed ↓"),
                       (axes[2], "legit_success", "(c) Real onboarding completed ↑")]:
    bars(ax, G, M, lambda m, c: E3[m].get(c, {}).get(key), GL, annotate=True, err=False, rot=True)
    ylim130(ax); ax.set_title(title)
axes[0].set_ylabel("Trials (%)")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.93), fontsize=6.2)
for ax in axes: ax.set_ylim(0, 112)
save(fig, "fig_gates")

# Fig 5
fig, ax = plt.subplots(figsize=(3.33, 2.05))
rows = [(m, c) for m in M for c in ["msg", "ctx"]]
for i, (m, c) in enumerate(rows):
    y = len(rows) - 1 - i
    o = 100*PP[m][c]["own_flag"][0]/PP[m][c]["own_flag"][1]; l = 100*PP[m][c]["look_flag"][0]/PP[m][c]["look_flag"][1]
    ax.plot([o, l], [y, y], color="#c3c2b7", lw=1.4, zorder=1)
    ax.scatter([o], [y], s=22, facecolor="white", edgecolor=COL[m], lw=1.2, zorder=3, marker="o")
    ax.scatter([l], [y], s=18, color=COL[m], zorder=2, marker="s")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{LAB[m]}, {'message only' if c=='msg' else '+ app. record'}" for m, c in rows][::-1], fontsize=5.8)
ax.set_xlim(-5, 105); ax.set_xlabel("Messages labelled SCAM (%), 24 identical-text pairs")
ax.set_ylim(-0.6, len(rows) + 0.9)
ax.legend([plt.Line2D([], [], ls="", marker="o", mfc="white", mec="#333"), plt.Line2D([], [], ls="", marker="s", color="#333")],
          ["Own-domain link (legitimate)", "Look-alike link (scam)"], frameon=False, fontsize=5.6, loc="upper center", ncol=2,
          bbox_to_anchor=(0.45, 1.0), handletextpad=0.3)
ax.spines[["top", "right"]].set_visible(False); ax.xaxis.grid(True, color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
save(fig, "fig_domain_blindness")
print(json.dumps(S["trials"]), "ok")
