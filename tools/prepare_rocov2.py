#!/usr/bin/env python
"""ROCOv2 (Ruckert et al., Scientific Data 2024) -> MMRL format
(task: retrieval, caption -> radiology image).

Uses the official train / valid / test splits, subsampled with a fixed seed.
Queries are captions (text); documents are images. Each split is ranked
against its own image pool (corpus_val.jsonl, corpus_test.jsonl); training
uses corpus.jsonl. The KG entity for SKGA is the first UMLS concept (CUI)
annotated for the image.

Input: the ROCOv2 release from Zenodo (record 10821435), unpacked so that
--root contains, per split, a captions CSV, a concepts CSV and an image folder.
File and column names are detected loosely (e.g. train_captions.csv with
columns ID,Caption; train_concepts.csv with ID,CUIs; cui_mapping.csv with
CUI,Canonical name; images in train/ or train_images/).

  python tools/prepare_rocov2.py --root /path/to/rocov2 --out data/rocov2
"""
import argparse
import glob
import json
import os
import random
import shutil

import pandas as pd

SPLITS = {"train": ["train"], "val": ["valid", "val", "validation"], "test": ["test"]}


def find(root, names, kind):
    for n in names:
        for pat in (f"{n}_{kind}*.csv", f"{n}*{kind}*.csv"):
            hits = sorted(glob.glob(os.path.join(root, "**", pat), recursive=True))
            hits = [h for h in hits if "manual" not in os.path.basename(h).lower()]
            if hits:
                return hits[0]
    return None


def image_index(root, names):
    """stem -> path on disk, or (zipfile, member) when images are still zipped (e.g. train_images.zip)."""
    import zipfile
    idx = {}
    for n in names:
        for d in glob.glob(os.path.join(root, "**", f"{n}*"), recursive=True):
            if os.path.isdir(d):
                for f in os.listdir(d):
                    stem, ext = os.path.splitext(f)
                    if ext.lower() in (".jpg", ".jpeg", ".png"):
                        idx[stem] = os.path.join(d, f)
            elif d.lower().endswith(".zip") and not idx:
                z = zipfile.ZipFile(d)
                for m in z.namelist():
                    stem, ext = os.path.splitext(os.path.basename(m))
                    if ext.lower() in (".jpg", ".jpeg", ".png"):
                        idx[stem] = (z, m)
        if idx:
            break
    return idx


def col(df, *keys):
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    raise KeyError(f"none of {keys} in {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_train", type=int, default=10000)
    ap.add_argument("--max_eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy_images", action="store_true", help="copy images instead of symlinking")
    ap.add_argument("--split_map", default="train=train,val=valid,test=test",
                    help="which official split feeds each of our splits; e.g. train=valid,val=valid,test=test "
                         "draws disjoint train and val sets from the official validation split")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "kg"), exist_ok=True)

    names = {}
    mp = sorted(glob.glob(os.path.join(a.root, "**", "*cui*mapping*.csv"), recursive=True))
    if mp:
        m = pd.read_csv(mp[0])
        names = dict(zip(m[col(m, "cui")].astype(str), m[m.columns[1]].astype(str)))

    smap = dict(kv.split("=") for kv in a.split_map.split(","))
    used = {}  # official split -> ids already taken (keeps train and val disjoint when they share a source)
    data, cuis = {}, set()
    for split in ("train", "val", "test"):
        src = smap[split]
        aliases = [src] + (["valid", "val", "validation"] if src in ("valid", "val", "validation") else [])
        cap_f, con_f = find(a.root, aliases, "caption"), find(a.root, aliases, "concept")
        if cap_f is None:
            raise FileNotFoundError(f"no captions CSV for {split} under {a.root}")
        caps = pd.read_csv(cap_f)
        caps = dict(zip(caps[col(caps, "id")].astype(str), caps[col(caps, "caption")].astype(str)))
        con = {}
        if con_f:
            c = pd.read_csv(con_f)
            for i, v in zip(c[col(c, "id")].astype(str), c[col(c, "cui")].astype(str)):
                con[i] = [x.strip() for x in v.replace(",", ";").split(";") if x.strip()]
        imgs = image_index(a.root, aliases)
        ids = sorted(i for i in caps if i in imgs and i not in used.get(src, set()))
        rng.shuffle(ids)
        ids = ids[: (a.max_train if split == "train" else a.max_eval)]
        used.setdefault(src, set()).update(ids)
        data[split] = (ids, caps, con, imgs)
        for i in ids:
            cuis.update(con.get(i, [])[:1])

    cui_list = sorted(cuis)
    cui_idx = {c: k for k, c in enumerate(cui_list)}
    stats = {}
    for split, (ids, caps, con, imgs) in data.items():
        queries, docs = [], []
        for i in ids:
            src = imgs[i]
            if isinstance(src, tuple):  # extract only the sampled images from the zip
                z, m = src
                dst = os.path.join(a.out, "images", os.path.basename(m))
                if not os.path.exists(dst):
                    with z.open(m) as fi, open(dst, "wb") as fo:
                        shutil.copyfileobj(fi, fo)
                src = dst
            dst = os.path.join(a.out, "images", os.path.basename(src))
            if not os.path.exists(dst):
                if a.copy_images:
                    shutil.copy(src, dst)
                else:
                    os.symlink(os.path.abspath(src), dst)
            ent = cui_idx.get((con.get(i) or [None])[0], -1)
            docs.append({"id": f"img_{i}", "image": f"images/{os.path.basename(src)}", "entity": ent})
            queries.append({"id": f"q_{i}", "text": caps[i], "entity": ent, "positives": [f"img_{i}"]})
        with open(os.path.join(a.out, f"{split}.jsonl"), "w") as f:
            for q in queries:
                f.write(json.dumps(q) + "\n")
        cname = "corpus.jsonl" if split == "train" else f"corpus_{split}.jsonl"
        with open(os.path.join(a.out, cname), "w") as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")
        stats[split] = len(ids)
    json.dump([names.get(c, c) for c in cui_list], open(os.path.join(a.out, "kg/entities.json"), "w"), indent=0)
    json.dump(cui_list, open(os.path.join(a.out, "kg/cuis.json"), "w"))
    meta = {"name": "rocov2", "task": "retrieval", "modalities": ["text", "image"], "max_text_len": 128,
            "entity_emb": "kg/entity_emb.npy", "entity_names": "kg/entities.json",
            "source": f"ROCOv2 (Ruckert et al., Sci. Data 2024), split map {a.split_map}, seed-{a.seed} subsample", "sizes": stats}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(stats), f"{len(cui_list)} entities")


if __name__ == "__main__":
    main()
