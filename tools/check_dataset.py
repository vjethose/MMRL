#!/usr/bin/env python
"""Validate a dataset directory before training: required files, missing
modality files, label range, class balance, split sizes and id overlap.

  python tools/check_dataset.py data/pubmed10k
"""
import json
import os
import sys
from collections import Counter


def main(root):
    meta = json.load(open(os.path.join(root, "meta.json")))
    ok = True
    for k in ("task", "modalities"):
        if k not in meta:
            print(f"meta.json is missing '{k}'"); ok = False
    if "graph" in meta.get("modalities", []) and "graph_in_dim" not in meta:
        print("meta.json needs graph_in_dim (and num_relations) when graphs are used"); ok = False
    ids = {}
    splits = ["train", "val", "test"] + (["corpus"] if meta.get("task") == "retrieval" else [])
    for s in splits:
        p = os.path.join(root, f"{s}.jsonl")
        if not os.path.exists(p):
            print(f"missing {s}.jsonl"); ok = False; continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        ids[s] = set(r["id"] for r in rows)
        miss, have = Counter(), Counter()
        for r in rows:
            for m in meta["modalities"]:
                v = r.get(m)
                if v is None:
                    continue
                have[m] += 1
                if m != "text" and not os.path.exists(v if os.path.isabs(v) else os.path.join(root, v)):
                    miss[m] += 1
        print(f"{s}: {len(rows)} rows; present per modality {dict(have)}; missing files {dict(miss) or 'none'}")
        if miss:
            ok = False
        if meta["task"] in ("classification", "vqa") and s != "corpus":
            n = meta.get("num_classes") or len(meta.get("answers", []))
            labels = Counter(r.get("label", -1) for r in rows)
            bad = [l for l in labels if not (0 <= l < n)]
            if bad:
                print(f"  labels outside [0, {n}): {bad[:5]}"); ok = False
            print(f"  class counts: {dict(sorted(labels.items()))}")
        ents = [r.get("entity", -1) for r in rows]
        print(f"  rows with a KG entity: {sum(e >= 0 for e in ents)}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in ids and b in ids and ids[a] & ids[b]:
            print(f"LEAK: {len(ids[a] & ids[b])} ids appear in both {a} and {b}"); ok = False
    ent = os.path.join(root, meta.get("entity_emb", "kg/entity_emb.npy"))
    if os.path.exists(ent):
        import numpy as np
        E = np.load(ent, mmap_mode="r")
        print(f"entity embeddings: {E.shape}")
    else:
        print("no entity embeddings found: SKGA will be disabled for this dataset")
    print("OK" if ok else "PROBLEMS FOUND")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
