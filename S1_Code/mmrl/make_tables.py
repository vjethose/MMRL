#!/usr/bin/env python
"""Aggregate run JSONs into the paper's tables and figures.

  python make_tables.py --results results --ablation_dataset pubmed10k \
      --roc_dataset arxiv_mml --retrieval_dataset biomed_retrieval --out paper_out

Writes paper_out/tables.md, paper_out/summary.json and paper_out/fig*.png.
Every number is mean ± SD over the seeds found in --results.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ORDER = ["unimodal", "early", "late", "mmbert", "cma", "mmrl"]
NAMES = {"unimodal": "UniModal", "early": "Early Fusion", "late": "Late Fusion", "mmbert": "MM-BERT",
         "cma": "Cross-Modal Attn.", "mmrl": "MMRL (Proposed)"}
MODS = ["text", "image", "audio", "graph"]


def ms(v):
    v = np.asarray(v, dtype=float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v))


def fmt(t, d=1):
    return f"{t[0]:.{d}f} ± {t[1]:.{d}f}" if t[2] > 1 else f"{t[0]:.{d}f}"


def config_key(r):
    a = r["args"]
    if a["method"] == "unimodal":
        return f"unimodal_{a['unimodal']}"
    if a["method"] != "mmrl":
        return a["method"] if not (a.get("ablate") or a.get("modalities") or a.get("tag")) else None
    if a.get("ablate"):
        return "mmrl_wo_" + a["ablate"]
    if a.get("modalities"):
        missing = [m for m in r.get("all_modalities", MODS) if m not in a["modalities"].split(",")]
        return "mmrl_wo_" + "_".join(missing) + "_enc" if missing else "mmrl"
    if a["lambda_align"] == 0:
        return "mmrl_wo_skga"
    return "mmrl" if not a.get("tag") else None


def load(results):
    runs = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(results, "*.json"))):
        r = json.load(open(f))
        if "test" not in r:
            continue
        meta_path = os.path.join(r["args"]["data"], "meta.json")
        if os.path.exists(meta_path):
            r["all_modalities"] = json.load(open(meta_path))["modalities"]
        k = config_key(r)
        if k:
            runs[r["dataset"]][k].append(r)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--cost", default=None, help="dir with profile_cost.py JSONs (default results/cost)")
    ap.add_argument("--ablation_dataset", default=None)
    ap.add_argument("--roc_dataset", default=None)
    ap.add_argument("--retrieval_dataset", default=None)
    ap.add_argument("--curve_dataset", default=None)
    ap.add_argument("--confusion_dataset", default=None, help="default: --roc_dataset, else the ablation dataset")
    ap.add_argument("--out", default="paper_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load(a.results)
    datasets = sorted(runs)
    S = {"datasets": datasets}
    md = []

    # ------------------------------------------------------------ Table IV
    table4 = {}
    uni_choice = {}
    for ds in datasets:
        R = runs[ds]
        uni = {k: v for k, v in R.items() if k.startswith("unimodal_")}
        if uni:
            best = max(uni, key=lambda k: np.mean([max(r["history"]["val"]) for r in uni[k]]))
            uni_choice[ds] = best.split("_", 1)[1]
            R = dict(R, unimodal=uni[best])
        table4[ds] = {m: ms([r["test"]["accuracy"] for r in R[m]]) for m in ORDER if m in R}
    S["table4"] = table4
    S["unimodal_choice"] = uni_choice
    md.append("## Table IV  Main metric (accuracy %; mAP@10 % for retrieval; EM % for VQA)\n")
    md.append("| Method | " + " | ".join(datasets) + " |")
    md.append("|---" * (len(datasets) + 1) + "|")
    for m in ORDER:
        md.append(f"| {NAMES[m]} | " + " | ".join(fmt(table4[d][m]) if m in table4[d] else "-" for d in datasets) + " |")
    gains = {}
    for d in datasets:
        base = [v[0] for k, v in table4[d].items() if k != "mmrl"]
        if "mmrl" in table4[d] and base:
            strongest = max((k for k in table4[d] if k != "mmrl"), key=lambda k: table4[d][k][0])
            gains[d] = {"vs": strongest, "gain": table4[d]["mmrl"][0] - table4[d][strongest][0]}
    S["gain_over_strongest"] = gains
    md.append("| Gain (pts) | " + " | ".join(f"{gains[d]['gain']:+.1f} vs {NAMES[gains[d]['vs']]}" if d in gains else "-" for d in datasets) + " |")
    md.append(f"\nUniModal row uses the best single modality by validation score: {uni_choice}\n")

    # ------------------------------------------------------------ Table V (ablation)
    abl_ds = a.ablation_dataset or (datasets[0] if datasets else None)
    if abl_ds:
        R = runs[abl_ds]
        rows = [("MMRL (Full)", "mmrl")] + [(f"w/o {m.capitalize()} Encoder", f"mmrl_wo_{m}_enc") for m in MODS] + [
            ("w/o Cross-Modal Fusion", "mmrl_wo_fusion"), ("w/o Attention Mechanism", "mmrl_wo_attention"),
            ("w/o MAFE (shared projection)", "mmrl_wo_mafe"), ("w/o SKGA (λ = 0)", "mmrl_wo_skga")]
        full = ms([r["test"]["accuracy"] for r in R.get("mmrl", [])]) if R.get("mmrl") else None
        t5 = {}
        md.append(f"## Table V  Ablation on {abl_ds}\n")
        md.append("| Configuration | Accuracy | F1 | mAP | Δ Acc |")
        md.append("|---|---|---|---|---|")
        for label, k in rows:
            if k not in R:
                continue
            acc = ms([r["test"]["accuracy"] for r in R[k]])
            f1 = ms([r["test"].get("f1", np.nan) for r in R[k]])
            mp = ms([r["test"].get("map", r["test"].get("map@10", np.nan)) for r in R[k]])
            delta = acc[0] - full[0] if (full and k != "mmrl") else None
            t5[k] = {"label": label, "acc": acc, "f1": f1, "map": mp, "delta": delta}
            md.append(f"| {label} | {fmt(acc)} | {fmt(f1)} | {fmt(mp)} | {'-' if delta is None else f'{delta:+.1f}'} |")
        S["table5"] = {"dataset": abl_ds, "rows": t5}
        md.append("")

    # ------------------------------------------------------------ Table VI (cost)
    cost_dir = a.cost or os.path.join(a.results, "cost")
    cost = defaultdict(list)
    for f in glob.glob(os.path.join(cost_dir, "*.json")):
        c = json.load(open(f))
        if c.get("ablate"):
            continue
        cost[c["method"]].append(c)
    if cost:
        md.append("## Table VI  Cost (batch 32, single GPU, data loading excluded)\n")
        md.append("| Method | Latency ms/batch | ms/sample | Peak GPU MB | Params M | GFLOPs/sample | Dataset |")
        md.append("|---|---|---|---|---|---|---|")
        t6 = {}
        for m in ORDER:
            if m not in cost:
                continue
            c = sorted(cost[m], key=lambda c: c["dataset"])[0] if m != "unimodal" else max(cost[m], key=lambda c: c["params_M"])
            t6[m] = c
            md.append(f"| {NAMES[m]} | {c['latency_ms_per_batch']:.1f} | {c['latency_ms_per_sample']:.2f} | {c['peak_mem_MB']:.0f} | "
                      f"{c['params_M']:.1f} | {c['gflops_per_sample']:.1f} | {c['dataset']} |")
        if "mmrl" in t6 and "cma" in t6:
            S["cost_vs_cma"] = {k: 100 * (1 - t6["mmrl"][k] / t6["cma"][k]) for k in
                                ("latency_ms_per_batch", "peak_mem_MB", "gflops_per_sample", "params_M") if t6["cma"][k]}
            md.append(f"\nMMRL vs CMA, % reduction: {json.dumps({k: round(v, 1) for k, v in S['cost_vs_cma'].items()})}")
        S["table6"] = t6
        md.append("")

    # ------------------------------------------------------------ convergence
    curve_ds = a.curve_dataset or abl_ds
    if curve_ds:
        conv = {}
        for m in ORDER:
            rs = runs[curve_ds].get(m) or []
            if rs:
                conv[m] = {"best_epoch": ms([r["history"]["best_epoch"] for r in rs]),
                           "epochs_run": ms([r["history"]["epochs_run"] for r in rs]),
                           "final_train_loss": ms([r["history"]["train_loss"][r["history"]["best_epoch"] - 1] for r in rs])}
        S["convergence"] = {"dataset": curve_ds, "methods": conv}
        md.append(f"## Convergence on {curve_ds} (epoch of best validation score)\n")
        for m, c in conv.items():
            md.append(f"- {NAMES[m]}: best epoch {fmt(c['best_epoch'])}, train loss there {fmt(c['final_train_loss'], 3)}")
        md.append("")

    # ------------------------------------------------------------ ROC / confusion
    roc_ds = a.roc_dataset
    if roc_ds and roc_ds in runs:
        R = runs[roc_ds]
        S["auc"] = {k: ms([r["test"]["auc"] for r in v]) for k, v in R.items() if (k == "mmrl" or k.startswith("unimodal_")) and "auc" in v[0]["test"]}
        md.append(f"## Macro AUC on {roc_ds}: " + ", ".join(f"{k} {fmt(v, 3)}" for k, v in S["auc"].items()) + "\n")
    conf_ds = a.confusion_dataset or (roc_ds if roc_ds in runs else abl_ds)
    if conf_ds and runs[conf_ds].get("mmrl") and "confusion" in runs[conf_ds]["mmrl"][0]["test"]:
        cm = np.array(runs[conf_ds]["mmrl"][0]["test"]["confusion"])
        off = cm.copy()
        np.fill_diagonal(off, 0)
        top = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:3], off.shape))[0]
        names = json.load(open(os.path.join(runs[conf_ds]["mmrl"][0]["args"]["data"], "meta.json"))).get("label_names")
        S["confusion"] = {"dataset": conf_ds, "seed": runs[conf_ds]["mmrl"][0]["args"]["seed"], "n": int(cm.sum()),
                          "per_class_correct": np.diag(cm).tolist(), "per_class_total": cm.sum(1).tolist(),
                          "top_confusions": [{"true": names[i] if names else int(i), "pred": names[j] if names else int(j), "count": int(cm[i, j])} for i, j in top]}

    # ------------------------------------------------------------ retrieval
    ret_ds = a.retrieval_dataset
    if ret_ds and ret_ds in runs:
        Ks = [1, 5, 10, 20, 50, 100]
        S["retrieval"] = {m: {K: ms([r["test"][f"map@{K}"] for r in runs[ret_ds][m]]) for K in Ks}
                          for m in ORDER if m in runs[ret_ds]}
        md.append(f"## mAP@K on {ret_ds}\n")
        md.append("| Method | " + " | ".join(f"K={K}" for K in Ks) + " |")
        md.append("|---" * (len(Ks) + 1) + "|")
        for m, v in S["retrieval"].items():
            md.append(f"| {NAMES[m]} | " + " | ".join(f"{v[K][0]:.1f}" for K in Ks) + " |")
        md.append("")

    # ------------------------------------------------------------ robustness, transfer, SKGA quartiles
    rob = {}
    for ds in datasets:
        for m in ("mmrl", "cma"):
            rs = [r for r in runs[ds].get(m, []) if "robustness" in r]
            if rs:
                rob.setdefault(ds, {})[m] = {mod: ms([r["robustness"][mod] for r in rs]) for mod in rs[0]["robustness"]}
    if rob:
        S["robustness"] = rob
        md.append("## Accuracy with one modality dropped at inference only\n")
        for ds, v in rob.items():
            for m, d in v.items():
                md.append(f"- {ds} {NAMES[m]}: " + ", ".join(f"no {k} {fmt(x)}" for k, x in d.items()))
        md.append("")
    tr = {ds: {m: ms([r["transfer"] for r in runs[ds][m] if "transfer" in r]) for m in runs[ds] if any("transfer" in r for r in runs[ds][m])} for ds in datasets}
    tr = {k: v for k, v in tr.items() if v}
    if tr:
        S["transfer"] = tr
    q = {ds: [r["align_quartiles"] for r in runs[ds].get("mmrl", []) if "align_quartiles" in r] for ds in datasets}
    q = {ds: {"top_acc": ms([x["top_acc"] for x in v]), "bottom_acc": ms([x["bottom_acc"] for x in v])} for ds, v in q.items() if v}
    if q:
        S["align_quartiles"] = q

    # ------------------------------------------------------------ figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
        col = {"unimodal": "#9aa5b1", "early": "#5c6b7a", "late": "#d4a017", "mmbert": "#b5651d", "cma": "#5b8c5a", "mmrl": "#3b5998"}
        # Fig 2: accuracy bars
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ms_ = [m for m in ORDER if any(m in table4[d] for d in datasets)]
        w = 0.8 / max(1, len(ms_))
        for i, m in enumerate(ms_):
            vals = [table4[d].get(m, (np.nan, 0, 0)) for d in datasets]
            ax.bar(np.arange(len(datasets)) + i * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals], label=NAMES[m], color=col[m], capsize=2)
        DSN = {"scienceqa": "ScienceQA (Acc)", "ogbn_arxiv": "ogbn-arxiv (Acc)", "rocov2": "ROCOv2 (mAP@10)"}
        ax.set_xticks(np.arange(len(datasets)) + w * (len(ms_) - 1) / 2, [DSN.get(d, d) for d in datasets])
        ax.set_ylabel("Score (%)")
        ax.legend(ncol=3, fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig2_accuracy.png"), dpi=300); plt.close(fig)
        # Fig 3: loss curves
        if curve_ds:
            fig, ax = plt.subplots(figsize=(7, 3))
            for m in ORDER:
                rs = runs[curve_ds].get(m if m != "unimodal" else f"unimodal_{uni_choice.get(curve_ds, 'text')}") or []
                if rs:
                    L = min(len(r["history"]["train_loss"]) for r in rs)
                    y = np.mean([r["history"]["train_loss"][:L] for r in rs], 0)
                    ax.plot(np.arange(1, L + 1), y, label=NAMES[m], color=col[m], lw=2 if m == "mmrl" else 1.2)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss"); ax.legend(fontsize=7, frameon=False)
            fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig3_loss.png"), dpi=300); plt.close(fig)
        # Fig 4: ROC
        if roc_ds and roc_ds in runs:
            fig, ax = plt.subplots(figsize=(3.6, 3.4))
            for k, v in runs[roc_ds].items():
                if (k == "mmrl" or k.startswith("unimodal_")) and "roc_curve" in v[0]["test"]:
                    rc = v[0]["test"]["roc_curve"]
                    lab = "MMRL" if k == "mmrl" else k.split("_")[1] + "-only"
                    ax.plot(rc["fpr"], rc["tpr"], label=f"{lab} (AUC {S['auc'][k][0]:.3f})", lw=2 if k == "mmrl" else 1)
            ax.plot([0, 1], [0, 1], "k--", lw=0.6)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=6, frameon=False)
            fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig4_roc.png"), dpi=300); plt.close(fig)
        # Fig 5: confusion
        if "confusion" in S:
            cm = np.array(runs[conf_ds]["mmrl"][0]["test"]["confusion"])
            names = json.load(open(os.path.join(runs[conf_ds]["mmrl"][0]["args"]["data"], "meta.json"))).get("label_names") or list(range(len(cm)))
            names = [n.replace("cs ", "cs.").upper().replace("CS.", "cs.") if isinstance(n, str) else n for n in names]
            if len(cm) <= 12:
                fig, ax = plt.subplots(figsize=(3.6, 3.4))
                ax.imshow(cm, cmap="Blues")
                for i in range(len(cm)):
                    for j in range(len(cm)):
                        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6, color="white" if cm[i, j] > cm.max() / 2 else "black")
                ax.set_xticks(range(len(cm)), names, rotation=45, ha="right", fontsize=6); ax.set_yticks(range(len(cm)), names, fontsize=6)
            else:
                # many classes: row-normalised (recall) heatmap, no per-cell numbers
                rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
                fig, ax = plt.subplots(figsize=(7, 6.4))
                im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
                ax.set_xticks(range(len(cm)), names, rotation=90, fontsize=5); ax.set_yticks(range(len(cm)), names, fontsize=5)
                cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Fraction of true class", fontsize=7); cb.ax.tick_params(labelsize=6)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig5_confusion.png"), dpi=300); plt.close(fig)
        # Fig 6: ablation
        if S.get("table5", {}).get("rows"):
            rows = S["table5"]["rows"]
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(range(len(rows)), [v["acc"][0] for v in rows.values()], yerr=[v["acc"][1] for v in rows.values()], color="#3b5998", capsize=2)
            ax.set_xticks(range(len(rows)), [v["label"].replace(" ", "\n", 1) for v in rows.values()], fontsize=6)
            ax.set_ylabel("Accuracy (%)")
            lo = min(v["acc"][0] for v in rows.values())
            ax.set_ylim(max(0, lo - 10), 100)
            fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig6_ablation.png"), dpi=300); plt.close(fig)
        # Fig 7: cost
        if cost:
            t6 = S["table6"]
            fig, axs = plt.subplots(1, 2, figsize=(7, 2.6))
            ks = list(t6)
            axs[0].bar(range(len(ks)), [t6[k]["latency_ms_per_batch"] for k in ks], color=[col[k] for k in ks]); axs[0].set_ylabel("Latency (ms / batch of 32)")
            axs[1].bar(range(len(ks)), [t6[k]["peak_mem_MB"] for k in ks], color=[col[k] for k in ks]); axs[1].set_ylabel("Peak GPU memory (MB)")
            for ax_ in axs:
                ax_.set_xticks(range(len(ks)), [NAMES[k] for k in ks], rotation=30, ha="right", fontsize=6)
            fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig7_cost.png"), dpi=300); plt.close(fig)
        # Fig 8: retrieval
        if "retrieval" in S:
            fig, ax = plt.subplots(figsize=(7, 3.2))
            for m, v in S["retrieval"].items():
                ax.plot(Ks, [v[K][0] for K in Ks], marker="o", label=NAMES[m], color=col[m], lw=2 if m == "mmrl" else 1.2)
            ax.set_xscale("log"); ax.set_xticks(Ks, Ks); ax.set_xlabel("K"); ax.set_ylabel("mAP@K (%)")
            ax.legend(fontsize=7, frameon=False)
            fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig8_retrieval.png"), dpi=300); plt.close(fig)
    except ImportError:
        print("matplotlib not installed; skipping figures")

    stats_path = os.path.join(a.results, "stats.json")
    if os.path.exists(stats_path):
        S["stats"] = json.load(open(stats_path))
    open(os.path.join(a.out, "tables.md"), "w").write("\n".join(md))
    json.dump(S, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
