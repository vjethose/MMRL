#!/usr/bin/env python
"""Build kg/entity_emb.npy for SKGA from entity names (kg/entities.json).

Each entity (a ScienceQA skill, a UMLS concept name, ...) is embedded once with
the frozen SciBERT encoder (mean of the last hidden layer). These vectors are
fixed during training; SKGA learns a linear map from them to width D.

  python tools/embed_entities.py data/scienceqa
  python tools/embed_entities.py data/rocov2 --random   # offline smoke test only
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--random", action="store_true", help="random vectors (no download) for smoke tests")
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    names = json.load(open(os.path.join(a.data, meta.get("entity_names", "kg/entities.json"))))
    out = os.path.join(a.data, meta.get("entity_emb", "kg/entity_emb.npy"))
    if a.random:
        E = np.random.default_rng(0).normal(0, 1, (len(names), 768)).astype(np.float32)
    else:
        import torch
        from transformers import AutoModel, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(a.model)
        m = AutoModel.from_pretrained(a.model).to(dev).eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(names), a.bs):
                b = tok(names[i:i + a.bs], padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
                h = m(**b).last_hidden_state
                w = b["attention_mask"].unsqueeze(-1).float()
                chunks.append(((h * w).sum(1) / w.sum(1)).cpu().numpy())
        E = np.concatenate(chunks).astype(np.float32)
    np.save(out, E)
    print(f"{out}: {E.shape}")


if __name__ == "__main__":
    main()
