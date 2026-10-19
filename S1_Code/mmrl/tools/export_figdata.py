#!/usr/bin/env python
"""Pack everything make_tables.py reads into one small JSON (figdata.json).

Keeps, per run JSON: dataset, the args make_tables uses, the test metrics it
reads, the training history, robustness / transfer / SKGA quartiles. Large
fields (ROC curve, confusion matrix) are kept only where a figure uses them.
Also packs the cost JSONs, stats.json and each dataset's meta.json.

  python tools/export_figdata.py --results results --data data --out figdata.json
  python tools/export_figdata.py --unpack figdata.json --to rebuilt   # then run make_tables on rebuilt/results
"""
import argparse
import glob
import json
import os

ARGS = ("method", "unimodal", "ablate", "modalities", "tag", "lambda_align", "seed", "data")
TEST = ("accuracy", "f1", "map", "auc", "auc_classes", "em") + tuple(f"{p}@{k}" for p in ("map", "recall") for k in (1, 5, 10, 20, 50, 100))


def rnd(x, d=4):
    if isinstance(x, float):
        return round(x, d)
    if isinstance(x, list):
        return [rnd(v, d) for v in x]
    if isinstance(x, dict):
        return {k: rnd(v, d) for k, v in x.items()}
    return x


def pack(a):
    runs = []
    for f in sorted(glob.glob(os.path.join(a.results, "*.json"))):
        r = json.load(open(f))
        if "test" not in r:
            continue
        ds = r["dataset"]
        args = {k: r["args"].get(k) for k in ARGS if r["args"].get(k) not in (None, "", [])}
        args["data"] = os.path.basename(os.path.normpath(r["args"]["data"]))
        if "lambda_align" not in args:
            args["lambda_align"] = r["args"].get("lambda_align")
        t = {k: r["test"][k] for k in TEST if k in r["test"]}
        m = r["args"]["method"]
        pf = f.replace(".json", "_preds.npz")
        if r["test"].get("auc") is not None and r["test"]["auc"] != r["test"]["auc"] and os.path.exists(pf):
            # result written by the older metric code (AUC over all classes fails when a class is absent from the
            # sampled test set): recompute AUC and the ROC curve from the saved per-example probabilities
            import numpy as np
            from sklearn.metrics import roc_auc_score, roc_curve
            from sklearn.preprocessing import label_binarize
            z = np.load(pf)
            P, y = z["probs"], z["label"]
            Y = label_binarize(y, classes=list(range(P.shape[1])))
            present = Y.sum(0) > 0
            t["auc"] = float(roc_auc_score(Y[:, present], P[:, present], average="macro"))
            t["auc_classes"] = int(present.sum())
            t["auc_recomputed_from_preds"] = True
            fpr, tpr, _ = roc_curve(Y.ravel(), P.ravel())
            idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 200)).astype(int)
            r["test"]["roc_curve"] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}
        plain = not (r["args"].get("ablate") or r["args"].get("modalities") or r["args"].get("tag"))
        if ds == a.roc_dataset and plain and m in ("mmrl", "unimodal") and "roc_curve" in r["test"]:
            rc = r["test"]["roc_curve"]
            step = max(1, len(rc["fpr"]) // 100)
            t["roc_curve"] = {"fpr": rc["fpr"][::step] + rc["fpr"][-1:], "tpr": rc["tpr"][::step] + rc["tpr"][-1:]}
        if ds == a.confusion_dataset and plain and m == "mmrl" and "confusion" in r["test"]:
            t["confusion"] = r["test"]["confusion"]
        h = r.get("history", {})
        out = {"file": os.path.basename(f), "dataset": ds, "args": args, "test": t,
               "history": {k: h[k] for k in ("val", "train_loss", "best_epoch", "epochs_run") if k in h}}
        for k in ("robustness", "transfer", "align_quartiles"):
            if k in r:
                out[k] = r[k]
        runs.append(rnd(out))
    cost = {}
    for d in sorted(glob.glob(os.path.join(a.results, "cost*"))):
        if os.path.isdir(d):
            cost[os.path.basename(d)] = {os.path.basename(f): rnd(json.load(open(f)), 5) for f in sorted(glob.glob(d + "/*.json"))}
    meta = {}
    for f in glob.glob(os.path.join(a.data, "*", "meta.json")):
        mm = json.load(open(f))
        meta[os.path.basename(os.path.dirname(f))] = {k: mm[k] for k in ("modalities", "label_names", "task", "name", "sizes") if k in mm}
    stats = json.load(open(os.path.join(a.results, "stats.json"))) if os.path.exists(os.path.join(a.results, "stats.json")) else None
    S = {"runs": runs, "cost": cost, "meta": meta, "stats": rnd(stats) if stats else None}
    json.dump(S, open(a.out, "w"), separators=(",", ":"))
    print(len(runs), "runs,", sum(len(v) for v in cost.values()), "cost files ->", a.out, os.path.getsize(a.out), "bytes")


def unpack(a):
    S = json.load(open(a.unpack))
    res, data = os.path.join(a.to, "results"), os.path.join(a.to, "data")
    for ds, mm in S["meta"].items():
        os.makedirs(os.path.join(data, ds), exist_ok=True)
        json.dump(mm, open(os.path.join(data, ds, "meta.json"), "w"), indent=1)
    os.makedirs(res, exist_ok=True)
    for r in S["runs"]:
        r = dict(r)
        r["args"] = dict(r["args"], data=os.path.join(data, r["args"]["data"]))
        json.dump(r, open(os.path.join(res, r.pop("file")), "w"))
    for d, files in S["cost"].items():
        os.makedirs(os.path.join(res, d), exist_ok=True)
        for n, c in files.items():
            json.dump(c, open(os.path.join(res, d, n), "w"))
    if S.get("stats"):
        json.dump(S["stats"], open(os.path.join(res, "stats.json"), "w"), indent=1)
    print("unpacked", len(S["runs"]), "runs into", a.to)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--data", default="data")
    p.add_argument("--roc_dataset", default="ogbn_arxiv")
    p.add_argument("--confusion_dataset", default="ogbn_arxiv")
    p.add_argument("--out", default="figdata.json")
    p.add_argument("--unpack", default=None)
    p.add_argument("--to", default="rebuilt")
    a = p.parse_args()
    unpack(a) if a.unpack else pack(a)
