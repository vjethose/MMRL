#!/usr/bin/env python
"""ogbn-arxiv (Hu et al., NeurIPS 2020) -> MMRL format
(task: classification into 40 arXiv CS categories, text + citation graph).

Uses the official time-based split, subsampled with a fixed seed. For each
paper: text = title + abstract; graph = the paper plus up to --k citation
neighbours (induced subgraph, OGB 128-d node features, relation 0 = cites,
1 = cited by). The centre paper is node 0. No KG entities (SKGA is off for
this dataset).

Inputs:
  --ogb_root  the ogbn_arxiv folder created by `ogb` (contains raw/, split/time/, mapping/)
  --titleabs  titleabs.tsv(.gz) from snap.stanford.edu/ogb/data/misc/ogbn_arxiv/

  python tools/prepare_ogbn_arxiv.py --ogb_root dataset/ogbn_arxiv --titleabs titleabs.tsv.gz --out data/ogbn_arxiv
"""
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd


def rd(path, **kw):
    return pd.read_csv(path, header=None, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ogb_root", required=True)
    ap.add_argument("--titleabs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=31, help="max neighbours per ego graph")
    ap.add_argument("--max_train", type=int, default=20000)
    ap.add_argument("--max_eval", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    R = a.ogb_root
    os.makedirs(os.path.join(a.out, "graphs"), exist_ok=True)

    feat = rd(os.path.join(R, "raw", "node-feat.csv.gz")).values.astype(np.float32)
    label = rd(os.path.join(R, "raw", "node-label.csv.gz")).values[:, 0].astype(int)
    edges = rd(os.path.join(R, "raw", "edge.csv.gz")).values.astype(int)
    out_n, in_n = defaultdict(list), defaultdict(list)
    for s, t in edges:
        out_n[s].append(t)
        in_n[t].append(s)
    paper = rd(os.path.join(R, "mapping", "nodeidx2paperid.csv.gz"), skiprows=1).values[:, 1].astype(int)
    ta = pd.read_csv(a.titleabs, sep="\t", header=None, names=["pid", "title", "abs"], quoting=3, dtype={"pid": str})
    ta = ta[pd.to_numeric(ta["pid"], errors="coerce").notna()]
    text = dict(zip(ta["pid"].astype(int), (ta["title"].fillna("") + ". " + ta["abs"].fillna("")).astype(str)))
    names = None
    lm = os.path.join(R, "mapping", "labelidx2arxivcategeory.csv.gz")
    if os.path.exists(lm):
        m = pd.read_csv(lm)
        names = [str(x).replace("arxiv ", "").strip() for x in m.iloc[:, 1]]

    stats = {}
    for split, fname, cap in (("train", "train", a.max_train), ("val", "valid", a.max_eval), ("test", "test", a.max_eval)):
        ids = rd(os.path.join(R, "split", "time", f"{fname}.csv.gz")).values[:, 0].astype(int).tolist()
        rng.shuffle(ids)
        ids = ids[:cap]
        rows = []
        for n in ids:
            nb = list(dict.fromkeys(out_n[n] + in_n[n]))
            rng.shuffle(nb)
            nodes = [n] + nb[: a.k]
            pos = {v: i for i, v in enumerate(nodes)}
            ei, et = [], []
            for v in nodes:
                for w in out_n[v]:
                    if w in pos:
                        ei += [(pos[v], pos[w]), (pos[w], pos[v])]
                        et += [0, 1]
            path = f"graphs/{n}.npz"
            np.savez_compressed(os.path.join(a.out, path), x=feat[nodes],
                                edge_index=np.array(ei, dtype=np.int64).T.reshape(2, -1), edge_type=np.array(et, dtype=np.int64))
            rows.append({"id": f"arxiv_{n}", "label": int(label[n]), "entity": -1,
                         "text": text.get(int(paper[n]), ""), "graph": path})
        with open(os.path.join(a.out, f"{split}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        stats[split] = len(rows)
    meta = {"name": "ogbn_arxiv", "task": "classification", "modalities": ["text", "graph"], "num_classes": int(label.max() + 1),
            "label_names": names, "graph_in_dim": int(feat.shape[1]), "num_relations": 2, "max_graph_nodes": a.k + 1,
            "max_text_len": 256, "source": f"ogbn-arxiv (Hu et al., NeurIPS 2020), official time split, seed-{a.seed} subsample",
            "sizes": stats}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
