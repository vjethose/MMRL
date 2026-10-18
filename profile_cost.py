#!/usr/bin/env python
"""Measure the cost numbers for Table VI: inference latency, peak GPU memory,
parameters and FLOPs, on a full-length synthetic batch (all modalities present,
max sequence lengths), single GPU, model in eval mode, data loading excluded.

  python profile_cost.py --data data/arxiv_mml --method mmrl
  python profile_cost.py --data data/arxiv_mml --method cma --bs 32 --iters 100
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from mmrl.data import AUDIO_HOP, Collator, HashTokenizer, HFTokenizer, load_meta, to_device
from mmrl.encoders import PRESETS
from mmrl.models import TaskModel


def fake_items(meta, mods, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    items = []
    for i in range(n):
        it = {"id": f"x{i}", "label": 0, "entity": 0}
        for m in mods:
            if m == "text":
                it[m] = " ".join(f"tok{j}" for j in range(meta["max_text_len"] + 8))
            elif m == "image":
                it[m] = torch.randn(3, meta["image_size"], meta["image_size"], generator=g)
            elif m == "audio":
                it[m] = torch.randn(meta["max_audio_tokens"] * AUDIO_HOP, generator=g)
            elif m == "graph":
                N = meta["max_graph_nodes"]
                ei = torch.randint(0, N, (2, 3 * N), generator=g)
                et = torch.randint(0, max(1, meta.get("num_relations", 1)), (3 * N,), generator=g)
                it[m] = (torch.randn(N, meta["graph_in_dim"], generator=g), ei, et)
        items.append(it)
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--unimodal", default=None)
    p.add_argument("--modalities", default=None)
    p.add_argument("--ablate", default="")
    p.add_argument("--preset", default="paper")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--amp", nargs="?", const="bf16", default="none", choices=["none", "bf16", "fp16"])
    p.add_argument("--out", default="results/cost")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_meta(a.data)
    mods = [a.unimodal] if a.method == "unimodal" else (a.modalities.split(",") if a.modalities else meta["modalities"])
    cfg = PRESETS[a.preset]
    tok = HFTokenizer(cfg["text_model"]) if cfg["text_model"] else HashTokenizer()
    ent_path = os.path.join(a.data, meta.get("entity_emb", "kg/entity_emb.npy"))
    ent = np.load(ent_path) if (a.method == "mmrl" and os.path.exists(ent_path)) else None
    model = TaskModel(a.method, meta, mods, a.preset, ablate=[x for x in a.ablate.split(",") if x],
                      unimodal=a.unimodal, entity_emb=ent).to(dev).eval()
    model.requires_grad_(False)  # inference only; also keeps FlopCounterMode's module tracker happy
    coll = Collator(meta, mods, tok)
    batch = to_device(coll(fake_items(meta, mods, a.bs)), dev)
    one = to_device(coll(fake_items(meta, mods, 1)), dev)
    ac = lambda: torch.autocast(dev.type, dtype=torch.float16 if a.amp == "fp16" else torch.bfloat16, enabled=(a.amp != "none" and dev.type == "cuda"))

    # parameters (SKGA entity table is a frozen buffer, not counted)
    params = sum(p.numel() for p in model.parameters()) / 1e6

    # FLOPs for one sample (forward only)
    with torch.no_grad(), ac():
        fc = FlopCounterMode(display=False)
        with fc:
            model(one)
    gflops = fc.get_total_flops() / 1e9

    # latency and peak memory at batch size a.bs
    with torch.no_grad(), ac():
        for _ in range(a.warmup):
            model(batch)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        times = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            model(batch)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        peak = (torch.cuda.max_memory_allocated() / 2**20) if dev.type == "cuda" else float("nan")
        act = ((torch.cuda.max_memory_allocated() - base) / 2**20) if dev.type == "cuda" else float("nan")

    times = np.array(times) * 1000
    res = {"dataset": os.path.basename(os.path.normpath(a.data)), "method": a.method, "unimodal": a.unimodal,
           "ablate": a.ablate, "modalities": mods, "device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
           "batch_size": a.bs, "latency_ms_per_batch": float(times.mean()), "latency_ms_per_batch_std": float(times.std()),
           "latency_ms_per_sample": float(times.mean() / a.bs), "peak_mem_MB": peak, "activation_mem_MB": act,
           "params_M": params, "gflops_per_sample": gflops, "amp": a.amp}
    os.makedirs(a.out, exist_ok=True)
    tag = "__".join(x for x in [res["dataset"], a.method, a.unimodal or "", a.ablate.replace(",", "_")] if x)
    with open(os.path.join(a.out, tag + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
