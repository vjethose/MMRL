#!/usr/bin/env python
"""Create a small synthetic dataset in the MMRL format, for smoke tests only.
Numbers obtained on it mean nothing for the paper.

  python tools/make_synthetic.py --out data/synth_cls --task classification
  python tools/make_synthetic.py --out data/synth_vqa --task vqa
  python tools/make_synthetic.py --out data/synth_ret --task retrieval
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

COLORS = [(220, 40, 40), (40, 160, 60), (40, 70, 200), (230, 180, 30), (150, 50, 180), (30, 170, 170)]


def text_for(c, rng, n_words=30, signal=0.25):
    vocab = [f"concept{c}_{i}" for i in range(10)]
    noise = [f"word{i}" for i in range(400)]
    return " ".join(rng.choice(vocab) if rng.random() < signal else rng.choice(noise) for _ in range(n_words))


def image_for(c, rng, path):
    im = Image.new("RGB", (64, 64), (255, 255, 255))
    d = ImageDraw.Draw(im)
    col = COLORS[c % len(COLORS)] if rng.random() < 0.7 else COLORS[rng.integers(len(COLORS))]
    x, y, r = rng.integers(10, 54), rng.integers(10, 54), rng.integers(6, 12)
    d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    arr = np.asarray(im).astype(np.float32) + rng.normal(0, 25, (64, 64, 3))
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def audio_for(c, rng, path, sr=16000):
    t = np.arange(int(sr * 1.5)) / sr
    f = 200 + 120 * c + rng.normal(0, 30)
    np.save(path, (0.5 * np.sin(2 * np.pi * f * t) + rng.normal(0, 0.5, t.shape)).astype(np.float32))


def graph_for(c, rng, path, F=16, R=3):
    n = int(rng.integers(8, 20))
    x = rng.normal(0, 1, (n, F)).astype(np.float32)
    x[:, c % F] += 0.8
    e = int(n * 2)
    ei = rng.integers(0, n, (2, e))
    et = rng.integers(0, R, e)
    np.savez(path, x=x, edge_index=ei, edge_type=et)


def sample(i, c, rng, root, mods, p_missing=0.1, prefix="s"):
    sid = f"{prefix}{i}"
    r = {"id": sid, "label": int(c), "entity": int(c)}
    for m in mods:
        if m != "text" and rng.random() < p_missing:
            r[m] = None
            continue
        if m == "text":
            r[m] = text_for(c, rng)
        elif m == "image":
            r[m] = f"images/{sid}.png"
            image_for(c, rng, os.path.join(root, r[m]))
        elif m == "audio":
            r[m] = f"audio/{sid}.npy"
            audio_for(c, rng, os.path.join(root, r[m]))
        elif m == "graph":
            r[m] = f"graphs/{sid}.npz"
            graph_for(c, rng, os.path.join(root, r[m]))
    return r


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="classification", choices=["classification", "vqa", "retrieval"])
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for d in ("images", "audio", "graphs", "kg"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    C = a.classes
    np.save(os.path.join(a.out, "kg/entity_emb.npy"), rng.normal(0, 1, (C + 14, 32)).astype(np.float32))
    meta = {"name": os.path.basename(a.out), "task": a.task, "graph_in_dim": 16, "num_relations": 3,
            "entity_emb": "kg/entity_emb.npy", "max_text_len": 64, "max_audio_tokens": 120}
    if a.task == "classification":
        mods = ["text", "image", "audio", "graph"]
        meta.update(modalities=mods, num_classes=C, label_names=[f"class{c}" for c in range(C)])
        n_val = a.n // 4
        for split, n, off in (("train", a.n, 0), ("val", n_val, 10**5), ("test", n_val, 2 * 10**5)):
            write(os.path.join(a.out, f"{split}.jsonl"), [sample(off + i, i % C, rng, a.out, mods) for i in range(n)])
    elif a.task == "vqa":
        mods = ["text", "image"]
        answers = [f"answer{c}" for c in range(C)]
        meta.update(modalities=mods, answers=answers)
        n_val = a.n // 4
        for split, n, off in (("train", a.n, 0), ("val", n_val, 10**5), ("test", n_val, 2 * 10**5)):
            rows = []
            for i in range(n):
                r = sample(off + i, i % C, rng, a.out, mods, p_missing=0.0)
                r["text"] = "what colour is the marker in the figure ? " + text_for(i % C, rng, 8, 0.2)
                rows.append(r)
            write(os.path.join(a.out, f"{split}.jsonl"), rows)
    else:
        mods = ["text", "image", "graph"]
        meta.update(modalities=mods)
        topics = 40
        docs = []
        for t in range(topics):
            for j in range(5):
                r = sample(t * 5 + j, t % C, rng, a.out, mods, prefix="doc")
                r["entity"] = t % (C + 14)
                r["text"] = f"topic{t} " * 3 + r["text"]
                r["topic"] = t
                docs.append(r)
        write(os.path.join(a.out, "corpus.jsonl"), docs)
        for split, n, off in (("train", 120, 0), ("val", 30, 10**5), ("test", 30, 2 * 10**5)):
            rows = []
            for i in range(n):
                t = int(rng.integers(topics))
                r = sample(off + i, t % C, rng, a.out, ["text", "image"], prefix="q")
                r["text"] = f"topic{t} " + text_for(t % C, rng, 10)
                r["entity"] = t % (C + 14)
                r["positives"] = [d["id"] for d in docs if d["topic"] == t]
                rows.append(r)
            write(os.path.join(a.out, f"{split}.jsonl"), rows)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
