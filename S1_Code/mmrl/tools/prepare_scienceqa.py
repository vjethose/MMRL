#!/usr/bin/env python
"""ScienceQA (Lu et al., NeurIPS 2022) -> MMRL format (task: vqa, text + image).

Uses the official train / validation / test splits. Keeps questions that have an
image (the IMG subset) unless --all is given. The answer is the index of the
correct option (at most 5 options); each sample stores n_choices so impossible
options are masked. Lecture and solution fields are NOT used (the solution
states the answer). The KG entity for SKGA is the question's `skill` label.

Input: the three parquet files of the Hugging Face dataset derek-thomas/ScienceQA
(download with `huggingface-cli download derek-thomas/ScienceQA --repo-type dataset`
or let --hf load it with the `datasets` library).

  python tools/prepare_scienceqa.py --src /path/to/ScienceQA/data --out data/scienceqa
"""
import argparse
import glob
import io
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

LETTERS = ["A", "B", "C", "D", "E"]


def load_split(src, name):
    files = sorted(glob.glob(os.path.join(src, "**", f"{name}-*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no {name}-*.parquet under {src}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def image_bytes(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, dict):
        return v.get("bytes")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with train-*.parquet, validation-*.parquet, test-*.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="also keep text-only questions (image marked missing)")
    ap.add_argument("--max_text_len", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "kg"), exist_ok=True)

    splits = {"train": load_split(a.src, "train"), "val": load_split(a.src, "validation"), "test": load_split(a.src, "test")}
    skills = sorted(set(s for df in splits.values() for s in df["skill"].astype(str)))
    sk_idx = {s: i for i, s in enumerate(skills)}
    stats = {}
    for split, df in splits.items():
        rows = []
        for i, r in df.iterrows():
            img = image_bytes(r.get("image"))
            if img is None and not a.all:
                continue
            choices = list(r["choices"])
            if len(choices) > len(LETTERS):
                continue
            sid = f"{split}_{i}"
            rec = {"id": sid, "label": int(r["answer"]), "n_choices": len(choices), "entity": sk_idx[str(r["skill"])],
                   "subject": str(r.get("subject", "")), "grade": str(r.get("grade", ""))}
            opts = " ".join(f"({LETTERS[k]}) {c}" for k, c in enumerate(choices))
            hint = str(r.get("hint") or "").strip()
            rec["options"] = [str(c) for c in choices]
            rec["text"] = f"Question: {r['question']} " + (f"Context: {hint} " if hint else "") + f"Options: {opts}"
            if img is not None:
                path = f"images/{sid}.png"
                Image.open(io.BytesIO(img)).convert("RGB").save(os.path.join(a.out, path))
                rec["image"] = path
            else:
                rec["image"] = None
            rows.append(rec)
        with open(os.path.join(a.out, f"{split}.jsonl"), "w") as f:
            for rec in rows:
                f.write(json.dumps(rec) + "\n")
        stats[split] = len(rows)
    json.dump(skills, open(os.path.join(a.out, "kg/entities.json"), "w"), indent=0)
    meta = {"name": "scienceqa", "task": "vqa", "multiple_choice": True, "max_option_len": 32, "modalities": ["text", "image"], "answers": LETTERS,
            "max_text_len": a.max_text_len, "entity_emb": "kg/entity_emb.npy", "entity_names": "kg/entities.json",
            "source": "ScienceQA (Lu et al., NeurIPS 2022), official splits, IMG subset" + (" + text-only" if a.all else ""),
            "sizes": stats}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(stats), f"{len(skills)} skills")


if __name__ == "__main__":
    main()
