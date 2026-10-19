#!/usr/bin/env python
"""Significance tests for Section V-H.

For every dataset, compares MMRL with each baseline:
  * paired t-test over seeds (pairs matched by seed), Bonferroni-corrected
    over all comparisons made on that dataset
  * Cohen's d for paired samples (mean difference / SD of differences)
  * McNemar's exact test on per-example correctness for each seed
    (classification and VQA only), with the largest corrected p reported

With 5 seeds the seed-level t-test has 4 degrees of freedom, so very small
p-values need very consistent gains; report the p-values you actually get.

  python stats.py --results results --out results/stats.json
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats


def load(results):
    runs = defaultdict(lambda: defaultdict(dict))  # dataset -> method_key -> seed -> run
    for f in glob.glob(os.path.join(results, "*.json")):
        r = json.load(open(f))
        if "test" not in r:
            continue
        a = r["args"]
        if a.get("ablate") or a.get("modalities") or a.get("tag") or (a["method"] == "mmrl" and a["lambda_align"] == 0):
            continue  # main comparison only
        key = a["method"] if a["method"] != "unimodal" else f"unimodal_{a['unimodal']}"
        runs[r["dataset"]][key][a["seed"]] = (r, f)
    return runs


def best_unimodal(methods):
    uni = {k: v for k, v in methods.items() if k.startswith("unimodal_")}
    if not uni:
        return methods
    # the paper's UniModal row = the best single modality, chosen by mean validation score
    best = max(uni, key=lambda k: np.mean([max(r["history"]["val"]) for r, _ in uni[k].values()]))
    out = {k: v for k, v in methods.items() if not k.startswith("unimodal_")}
    out["unimodal"] = uni[best]
    out["_unimodal_choice"] = best
    return out


def mcnemar(a, b):
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {}
    for ds, methods in load(a.results).items():
        methods = best_unimodal(methods)
        choice = methods.pop("_unimodal_choice", None)
        if "mmrl" not in methods:
            continue
        base = [m for m in methods if m != "mmrl"]
        k = len(base)
        out[ds] = {"unimodal_choice": choice, "comparisons": {}}
        for b in base:
            seeds = sorted(set(methods["mmrl"]) & set(methods[b]))
            x = np.array([methods["mmrl"][s][0]["test"]["accuracy"] for s in seeds])
            y = np.array([methods[b][s][0]["test"]["accuracy"] for s in seeds])
            diff = x - y
            res = {"n_seeds": len(seeds), "mmrl_mean": float(x.mean()), "baseline_mean": float(y.mean()),
                   "mean_diff": float(diff.mean())}
            if len(seeds) >= 2 and diff.std(ddof=1) > 0:
                t, p = stats.ttest_rel(x, y)
                res.update(t=float(t), p=float(p), p_bonferroni=float(min(1.0, p * k)),
                           cohens_d=float(diff.mean() / diff.std(ddof=1)))
            mc = []
            for s in seeds:
                fa = methods["mmrl"][s][1].replace(".json", "_preds.npz")
                fb = methods[b][s][1].replace(".json", "_preds.npz")
                if os.path.exists(fa) and os.path.exists(fb):
                    A, Bn = np.load(fa), np.load(fb)
                    ca = (A["probs"].argmax(1) == A["label"]).astype(int)
                    cb = (Bn["probs"].argmax(1) == Bn["label"]).astype(int)
                    mc.append(min(1.0, mcnemar(ca, cb) * k))
            if mc:
                res["mcnemar_p_bonferroni_max_over_seeds"] = float(max(mc))
            out[ds]["comparisons"][b] = res
    js = json.dumps(out, indent=1)
    print(js)
    if a.out:
        open(a.out, "w").write(js)


if __name__ == "__main__":
    main()
