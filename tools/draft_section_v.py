#!/usr/bin/env python
"""Draft the text of Section V from paper_out/summary.json.

The wording follows the numbers: if MMRL loses to a baseline, or an ablation
does not hurt, the text says so. Read and edit the output; do not paste it
blindly.

  python tools/draft_section_v.py paper_out/summary.json > paper_out/section_v.md
"""
import json
import sys

NAMES = {"unimodal": "UniModal", "early": "Early Fusion", "late": "Late Fusion", "mmbert": "MM-BERT",
         "cma": "Cross-Modal Attention", "mmrl": "MMRL"}


def f1(x):
    return f"{x:.1f}"


def main(path):
    S = json.load(open(path))
    out = []
    t4, gains = S.get("table4", {}), S.get("gain_over_strongest", {})
    ds = S.get("datasets", [])

    out.append("## A. Overall Performance Comparison\n")
    if gains:
        wins = [d for d in ds if d in gains and gains[d]["gain"] > 0]
        losses = [d for d in ds if d in gains and gains[d]["gain"] <= 0]
        g = [gains[d]["gain"] for d in wins]
        s = (f"Table IV compares MMRL with the baselines on all {len(ds)} datasets (mean ± SD over "
             f"{t4[ds[0]]['mmrl'][2] if ds and 'mmrl' in t4[ds[0]] else 'N'} seeds). ")
        if wins:
            s += (f"MMRL has the highest score on {len(wins)} of {len(ds)} datasets "
                  f"({', '.join(wins)}), with gains of {f1(min(g))} to {f1(max(g))} points over the strongest baseline on each. ")
        for d in losses:
            s += (f"On {d}, {NAMES[gains[d]['vs']]} scores {f1(-gains[d]['gain'])} points higher than MMRL. ")
        out.append(s)
        for d in ds:
            if d in gains:
                out.append(f"- {d}: MMRL {f1(t4[d]['mmrl'][0])} ± {f1(t4[d]['mmrl'][1])} vs {NAMES[gains[d]['vs']]} "
                           f"{f1(t4[d][gains[d]['vs']][0])} ± {f1(t4[d][gains[d]['vs']][1])}")
        if S.get("unimodal_choice"):
            out.append(f"\nThe UniModal row uses the best single modality on validation data: {S['unimodal_choice']}.")
    out.append("")

    c = S.get("convergence")
    if c:
        out.append("## B. Convergence Analysis\n")
        parts = [f"{NAMES[m]} reaches its best validation score at epoch {f1(v['best_epoch'][0])}" for m, v in c["methods"].items()]
        out.append(f"On {c['dataset']}, " + "; ".join(parts) + ". Training loss alone does not show better generalization, so we rely on the held-out results in Table IV.\n")

    if S.get("auc"):
        out.append("## C. ROC Curve Analysis\n")
        a = S["auc"]
        txt = ", ".join(f"{('MMRL' if k == 'mmrl' else k.split('_')[1] + '-only')} {v[0]:.3f}" for k, v in sorted(a.items(), key=lambda kv: -kv[1][0]))
        out.append(f"Macro-averaged one-vs-rest AUC: {txt}.\n")

    cm = S.get("confusion")
    if cm:
        out.append("## D. Confusion Matrix Analysis\n")
        corr = sum(cm["per_class_correct"])
        out.append(f"On the {cm['dataset']} test set (N = {cm['n']}, seed {cm['seed']}), MMRL classifies {corr} samples correctly "
                   f"({100 * corr / cm['n']:.1f}%). The largest off-diagonal counts are "
                   + ", ".join(f"{t['true']} predicted as {t['pred']} ({t['count']})" for t in cm["top_confusions"]) + ".\n")

    t5 = S.get("table5")
    if t5:
        out.append("## E. Ablation Study\n")
        rows = t5["rows"]
        full = rows.get("mmrl")
        out.append(f"Table V reports the ablation on {t5['dataset']}. The full model reaches {f1(full['acc'][0])}% accuracy." if full else "")
        drops = sorted([(v["delta"], v["label"]) for k, v in rows.items() if v.get("delta") is not None])
        for dlt, lab in drops:
            verb = "lowers" if dlt < 0 else "raises"
            out.append(f"- {lab} {verb} accuracy by {f1(abs(dlt))} points.")
        out.append("")

    t6 = S.get("table6")
    if t6:
        out.append("## F. Computational Efficiency\n")
        if S.get("cost_vs_cma"):
            r = S["cost_vs_cma"]
            def say(k, noun):
                v = r.get(k)
                return None if (v is None or v != v) else (f"{f1(abs(v))}% {'lower' if v > 0 else 'higher'} {noun}")
            parts = [p for p in (say("latency_ms_per_batch", "inference latency"), say("peak_mem_MB", "peak GPU memory"),
                                 say("gflops_per_sample", "FLOPs per sample"), say("params_M", "parameter count")) if p]
            m, c_ = t6["mmrl"], t6["cma"]
            out.append(f"Compared with Cross-Modal Attention, MMRL has {', '.join(parts)} "
                       f"({f1(m['latency_ms_per_batch'])} vs {f1(c_['latency_ms_per_batch'])} ms per batch of 32; "
                       f"{m['peak_mem_MB']:.0f} vs {c_['peak_mem_MB']:.0f} MB; {f1(m['gflops_per_sample'])} vs {f1(c_['gflops_per_sample'])} GFLOPs; "
                       f"{f1(m['params_M'])}M vs {f1(c_['params_M'])}M parameters), measured on {m['device']}.\n")

    r = S.get("retrieval")
    if r and "mmrl" in r:
        out.append("## G. Cross-Modal Retrieval Performance\n")
        best_b = max((m for m in r if m != "mmrl"), key=lambda m: r[m]["10"][0] if "10" in r[m] else r[m][10][0])
        g = lambda m, K: (r[m][str(K)] if str(K) in r[m] else r[m][K])[0]
        out.append(f"At K = 1, MMRL reaches {f1(g('mmrl', 1))} mAP against {f1(g(best_b, 1))} for {NAMES[best_b]}; "
                   f"at K = 10, {f1(g('mmrl', 10))} vs {f1(g(best_b, 10))}; at K = 100, {f1(g('mmrl', 100))} vs {f1(g(best_b, 100))}.\n")

    st = S.get("stats")
    if st:
        out.append("## H. Statistical Significance\n")
        for d, v in st.items():
            for b, c in v["comparisons"].items():
                if "p_bonferroni" in c:
                    out.append(f"- {d}, MMRL vs {b}: diff {c['mean_diff']:+.2f}, paired t p = {c['p_bonferroni']:.3g} (Bonferroni), "
                               f"Cohen's d = {c['cohens_d']:.2f}"
                               + (f", McNemar p ≤ {c['mcnemar_p_bonferroni_max_over_seeds']:.3g}" if "mcnemar_p_bonferroni_max_over_seeds" in c else ""))
        out.append("")

    if S.get("robustness") or S.get("align_quartiles") or S.get("transfer"):
        out.append("## Numbers for Section VI\n")
        for d, v in S.get("robustness", {}).items():
            for m, x in v.items():
                out.append(f"- {d} {NAMES[m]}, one modality dropped at test: " + ", ".join(f"no {k} {f1(y[0])}" for k, y in x.items()))
        for d, v in S.get("align_quartiles", {}).items():
            out.append(f"- {d} SKGA alignment quartiles: top {f1(v['top_acc'][0])}% vs bottom {f1(v['bottom_acc'][0])}% accuracy")
        for d, v in S.get("transfer", {}).items():
            out.append(f"- {d} transfer: " + ", ".join(f"{NAMES.get(k, k)} {f1(x[0])}" for k, x in v.items()))
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
