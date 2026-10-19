#!/usr/bin/env python
"""Train and evaluate one (dataset, method, seed) configuration.

Example:
  python train.py --data data/pubmed10k --method mmrl --seed 0
  python train.py --data data/pubmed10k --method mmrl --ablate mafe --seed 0
  python train.py --data data/pubmed10k --method mmrl --lambda_align 0 --tag wo_skga --seed 0
  python train.py --data data/pubmed10k --method unimodal --unimodal text --seed 0
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from mmrl.data import Collator, HashTokenizer, HFTokenizer, MMDataset, load_meta, read_jsonl, to_device
from mmrl.encoders import PRESETS
from mmrl.metrics import classification_metrics, retrieval_metrics, vqa_metrics
from mmrl.models import TaskModel


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--method", required=True, choices=["mmrl", "unimodal", "early", "late", "cma", "mmbert"])
    p.add_argument("--modalities", default=None, help="comma list; default = all modalities in meta.json")
    p.add_argument("--unimodal", default=None, help="modality for --method unimodal")
    p.add_argument("--ablate", default="", help="comma list from {mafe,fusion,attention}")
    p.add_argument("--lambda_align", type=float, default=0.1, help="SKGA weight; 0 = w/o SKGA")
    p.add_argument("--tau_align", type=float, default=0.07)
    p.add_argument("--tau_retrieval", type=float, default=0.05)
    p.add_argument("--preset", default="paper", choices=list(PRESETS))
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5, help="encoder learning rate")
    p.add_argument("--head_lr", type=float, default=1e-4, help="learning rate of fusion layers and heads")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", nargs="?", const="bf16", default="none", choices=["none", "bf16", "fp16"],
                   help="mixed precision; use fp16 on T4/P100/V100 (no fast bf16), bf16 on A100/H100")
    p.add_argument("--robustness", action="store_true", help="also test with each modality dropped at inference")
    p.add_argument("--transfer_data", default=None, help="extra test set (same label space) for zero-shot transfer")
    p.add_argument("--max_train", type=int, default=0, help="debug: cap training samples")
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="")
    p.add_argument("--save_ckpt", action="store_true")
    return p.parse_args(argv)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_name(a):
    parts = [os.path.basename(os.path.normpath(a.data)), a.method]
    if a.method == "unimodal":
        parts.append(a.unimodal)
    if a.ablate:
        parts.append("wo_" + a.ablate.replace(",", "_"))
    if a.modalities:
        parts.append("mods_" + a.modalities.replace(",", "_"))
    if a.method == "mmrl" and a.lambda_align == 0 and "wo_skga" not in a.tag:
        parts.append("wo_skga")
    if a.tag:
        parts.append(a.tag)
    parts.append(f"s{a.seed}")
    return "__".join(parts)


class Runner:
    def __init__(self, a):
        self.a = a
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta = load_meta(a.data)
        self.task = self.meta["task"]
        mods = a.modalities.split(",") if a.modalities else self.meta["modalities"]
        if a.method == "unimodal":
            mods = [a.unimodal]
        self.mods = [m for m in mods if m in self.meta["modalities"]]
        cfg = PRESETS[a.preset]
        self.tok = HFTokenizer(cfg["text_model"]) if cfg["text_model"] else HashTokenizer()
        ent = None
        ent_path = os.path.join(a.data, self.meta.get("entity_emb", "kg/entity_emb.npy"))
        if a.method == "mmrl" and a.lambda_align > 0 and os.path.exists(ent_path):
            ent = np.load(ent_path)
        self.model = TaskModel(a.method, self.meta, self.mods, a.preset, D=a.dim,
                               ablate=[x for x in a.ablate.split(",") if x], unimodal=a.unimodal,
                               entity_emb=ent, tau_align=a.tau_align).to(self.dev)
        self.collate = Collator(self.meta, self.mods, self.tok)
        if self.task == "retrieval":
            self.corpus = read_jsonl(os.path.join(a.data, "corpus.jsonl"))
            self.doc_index = {r["id"]: i for i, r in enumerate(self.corpus)}

    # ------------------------------------------------------------ data
    def loader(self, split=None, train=False, records=None, root=None, drop=()):
        ds = MMDataset(root or self.a.data, split, self.meta, self.mods, train=train, records=records, drop_at_test=drop)
        if train and self.a.max_train:
            ds.records = ds.records[: self.a.max_train]
        g = torch.Generator()
        g.manual_seed(self.a.seed)
        return torch.utils.data.DataLoader(ds, batch_size=self.a.bs, shuffle=train, num_workers=self.a.workers,
                                           collate_fn=self.collate, generator=g, drop_last=train and self.task == "retrieval",
                                           worker_init_fn=lambda w: np.random.seed(self.a.seed * 100 + w),
                                           persistent_workers=False, pin_memory=self.dev.type == "cuda")

    def autocast(self):
        on = self.a.amp != "none" and self.dev.type == "cuda"
        dt = torch.float16 if self.a.amp == "fp16" else torch.bfloat16
        return torch.autocast(self.dev.type, dtype=dt, enabled=on)

    @staticmethod
    def mask_choices(logits, batch):
        """ScienceQA-style multiple choice: answers beyond the question's own choices are impossible."""
        n = batch.get("n_choices")
        if n is None or logits is None or not bool((n > 0).any()):
            return logits
        idx = torch.arange(logits.shape[1], device=logits.device)[None]
        bad = (idx >= n[:, None]) & (n[:, None] > 0)
        return logits.float().masked_fill(bad, -1e4)

    # ------------------------------------------------------------ train
    def loss(self, batch):
        out = self.model(batch)
        if self.task == "retrieval":
            q = F.normalize(out["emb"], dim=-1)
            docs = [self.corpus[self.doc_index[random.choice(i)]] for i in batch["positives"]]
            dbatch = to_device(self.collate([self.doc_ds[self.doc_index[d["id"]]] for d in docs]), self.dev)
            dout = self.model(dbatch)
            k = F.normalize(dout["emb"], dim=-1)
            logits = q @ k.T / self.a.tau_retrieval
            tgt = torch.arange(len(q), device=self.dev)
            task = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
            align = self.model.align_loss(dout["emb"], dbatch["entity"]) if self.a.lambda_align > 0 else 0.0
        else:
            task = F.cross_entropy(self.mask_choices(out["logits"], batch).float(), batch["label"])
            align = self.model.align_loss(out["emb"], batch["entity"]) if self.a.lambda_align > 0 else 0.0
        return task + self.a.lambda_align * align, float(task.detach()), float(align.detach() if torch.is_tensor(align) else align)

    def fit(self):
        a = self.a
        tr = self.loader("train", train=True)
        if self.task == "retrieval":
            self.doc_ds = MMDataset(a.data, None, self.meta, self.mods, train=True, records=self.corpus)
            tr.dataset.records = [r for r in tr.dataset.records if r.get("positives")]
            # positives are attached to the batch by wrapping the collator
            base = self.collate
            pos_map = {r["id"]: [p for p in r["positives"] if p in self.doc_index] for r in tr.dataset.records}

            def coll(items):
                b = base(items)
                b["positives"] = [pos_map[i] for i in b["ids"]]
                return b
            tr.collate_fn = coll
        enc = set(id(p) for p in self.model.encoder_params())
        groups = [{"params": [p for p in self.model.parameters() if id(p) in enc], "lr": a.lr},
                  {"params": [p for p in self.model.parameters() if id(p) not in enc], "lr": a.head_lr}]
        opt = torch.optim.AdamW(groups, weight_decay=a.wd)
        steps = a.epochs * math.ceil(len(tr) / a.accum)
        warm = max(1, int(a.warmup * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

        scaler = torch.amp.GradScaler("cuda", enabled=(a.amp == "fp16" and self.dev.type == "cuda"))
        best, best_state, bad = -1e9, None, 0
        hist = {"train_loss": [], "task_loss": [], "align_loss": [], "val": [], "epoch_time_s": []}
        for ep in range(a.epochs):
            self.model.train()
            t0, tot, tt, ta, n = time.time(), 0.0, 0.0, 0.0, 0
            opt.zero_grad(set_to_none=True)
            for i, batch in enumerate(tr):
                positives = batch.pop("positives", None)
                batch = to_device(batch, self.dev)
                if positives is not None:
                    batch["positives"] = positives
                with self.autocast():
                    loss, lt, la = self.loss(batch)
                scaler.scale(loss / a.accum).backward()
                if (i + 1) % a.accum == 0 or i + 1 == len(tr):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                tot, tt, ta, n = tot + float(loss.detach()), tt + lt, ta + la, n + 1
            val = self.evaluate("val")["accuracy"]
            hist["train_loss"].append(tot / max(n, 1))
            hist["task_loss"].append(tt / max(n, 1))
            hist["align_loss"].append(ta / max(n, 1))
            hist["val"].append(val)
            hist["epoch_time_s"].append(time.time() - t0)
            print(f"epoch {ep + 1}: loss {tot / max(n, 1):.4f}  val {val:.2f}  ({time.time() - t0:.0f}s)", flush=True)
            if val > best:
                best, bad, hist["best_epoch"] = val, 0, ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        self.model.load_state_dict(best_state)
        hist["epochs_run"] = len(hist["train_loss"])
        return hist

    # ------------------------------------------------------------ evaluate
    @torch.no_grad()
    def embed(self, loader):
        self.model.eval()
        embs, probs, labels, ents, ids, alphas = [], [], [], [], [], []
        for batch in loader:
            ids += batch["ids"]
            batch = to_device(batch, self.dev)
            with self.autocast():
                out = self.model(batch)
            embs.append(out["emb"].float().cpu())
            if out.get("logits") is not None:
                probs.append(torch.softmax(self.mask_choices(out["logits"], batch).float(), -1).cpu())
            if out.get("alpha") is not None:
                alphas.append(out["alpha"].float().cpu())
            labels.append(batch["label"].cpu())
            ents.append(batch["entity"].cpu())
        cat = lambda x: torch.cat(x) if x else None
        return {"emb": cat(embs), "probs": cat(probs), "label": cat(labels), "entity": cat(ents), "ids": ids, "alpha": cat(alphas)}

    def evaluate(self, split, root=None, drop=(), keep=False):
        if self.task == "retrieval":
            base = root or self.a.data
            own = os.path.join(base, f"corpus_{split}.jsonl")  # rank each split against its own pool if given
            corpus = read_jsonl(own) if os.path.exists(own) else self.corpus
            index = {r["id"]: i for i, r in enumerate(corpus)}
            q = self.embed(self.loader(split, root=root, drop=drop))
            d = self.embed(self.loader(records=corpus, root=root, drop=drop))
            S = (F.normalize(q["emb"], dim=-1) @ F.normalize(d["emb"], dim=-1).T).numpy()
            recs = read_jsonl(os.path.join(base, f"{split}.jsonl"))
            rel = [set(index[p] for p in r.get("positives", []) if p in index) for r in recs]
            return retrieval_metrics(S, rel)
        r = self.embed(self.loader(split, root=root, drop=drop))
        P, y = r["probs"].numpy(), r["label"].numpy()
        m = classification_metrics(P, y, P.shape[1]) if self.task == "classification" else vqa_metrics(P, y)
        if keep:
            m["_raw"] = r
        return m


def main(argv=None):
    a = get_args(argv)
    set_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    name = run_name(a)
    R = Runner(a)
    n_params = sum(p.numel() for p in R.model.parameters())
    print(f"{name}: {n_params / 1e6:.1f}M parameters on {R.dev}", flush=True)
    t0 = time.time()
    hist = R.fit()
    train_time = time.time() - t0
    res = {"name": name, "args": vars(a), "dataset": os.path.basename(os.path.normpath(a.data)), "task": R.task,
           "method": a.method, "modalities": R.mods, "params_M": n_params / 1e6, "train_time_s": train_time,
           "history": hist}
    test = R.evaluate("test", keep=True)
    raw = test.pop("_raw", None)
    res["test"] = test
    if raw is not None:
        np.savez_compressed(os.path.join(a.out, name + "_preds.npz"), probs=raw["probs"].numpy(), label=raw["label"].numpy(),
                            ids=np.array(raw["ids"]))
        if R.model.entities is not None:  # post-hoc SKGA quartile analysis
            with torch.no_grad():
                score = R.model.align_score(raw["emb"].to(R.dev), raw["entity"].to(R.dev)).cpu().numpy()
            ok = ~np.isnan(score)
            if ok.sum() >= 8:
                correct = (raw["probs"].numpy().argmax(1) == raw["label"].numpy())[ok]
                qs = np.quantile(score[ok], [0.25, 0.75])
                res["align_quartiles"] = {"bottom_acc": 100 * float(correct[score[ok] <= qs[0]].mean()),
                                          "top_acc": 100 * float(correct[score[ok] >= qs[1]].mean())}
        if raw.get("alpha") is not None:
            res["mean_modality_weights"] = dict(zip(R.mods, raw["alpha"].mean(0).tolist())) if raw["alpha"].shape[1] == len(R.mods) else None
    if a.robustness and len(R.mods) > 1 and R.task != "retrieval":  # retrieval queries/docs carry one modality each
        res["robustness"] = {m: R.evaluate("test", drop=(m,))["accuracy"] for m in R.mods}
    if a.transfer_data:
        res["transfer"] = R.evaluate("test", root=a.transfer_data)["accuracy"]
    with open(os.path.join(a.out, name + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    if a.save_ckpt:
        torch.save(R.model.state_dict(), os.path.join(a.out, name + ".pt"))
    print(json.dumps({k: v for k, v in test.items() if k not in ("confusion", "roc_curve")}, indent=1))


if __name__ == "__main__":
    main()
