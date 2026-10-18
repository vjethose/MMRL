"""Dataset loading for MMRL.

Each dataset lives in one directory:

  meta.json            task description (see README)
  train.jsonl          one sample per line
  val.jsonl
  test.jsonl
  corpus.jsonl         retrieval only: documents to rank
  kg/entity_emb.npy    [num_entities, d_e] pre-trained KG entity embeddings (for SKGA)

A sample line looks like:
  {"id": "PMC123", "text": "...", "image": "images/PMC123.png",
   "audio": "audio/PMC123.npy", "graph": "graphs/PMC123.npz",
   "entity": 42, "label": 3}
Any modality key may be missing or null. Retrieval queries also carry
"positives": ["doc_id", ...].
"""
import json
import os
import random
import zlib

import numpy as np
import torch
from PIL import Image, ImageEnhance

MODALITIES = ("text", "image", "audio", "graph")
IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
AUDIO_HOP = 8 * 4 * 2 * 2  # total stride of the audio conv front end


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_meta(root):
    with open(os.path.join(root, "meta.json")) as f:
        meta = json.load(f)
    meta.setdefault("max_text_len", 512)
    meta.setdefault("image_size", 224)
    meta.setdefault("max_audio_tokens", 300)
    meta.setdefault("max_graph_nodes", 128)
    meta.setdefault("audio_sr", 16000)
    return meta


# ---------------------------------------------------------------- tokenizers
class HashTokenizer:
    """Offline stand-in for WordPiece, used only with --preset tiny."""

    def __init__(self, vocab_size=30522):
        self.vocab_size = vocab_size

    def __call__(self, texts, max_length):
        ids = np.zeros((len(texts), max_length), dtype=np.int64)
        mask = np.zeros((len(texts), max_length), dtype=np.int64)
        for i, t in enumerate(texts):
            toks = [101] + [1000 + (zlib.crc32(w.encode()) % (self.vocab_size - 1000)) for w in t.lower().split()]
            toks = toks[: max_length - 1] + [102]
            ids[i, : len(toks)] = toks
            mask[i, : len(toks)] = 1
        return {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask)}


class HFTokenizer:
    def __init__(self, name):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)

    def __call__(self, texts, max_length):
        out = self.tok(texts, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}


# ---------------------------------------------------------------- loaders
def load_image(path, size, train, rotate360):
    img = Image.open(path).convert("RGB")
    if train:
        if rotate360:
            img = img.rotate(random.uniform(0, 360), fillcolor=(255, 255, 255))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    img = img.resize((size, size), Image.BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 255.0
    if train:
        x = np.clip(x + np.random.normal(0, 0.02, x.shape).astype(np.float32), 0, 1)
    x = (x - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def load_audio(path, max_samples, train):
    if path.endswith(".npy"):
        wav = np.load(path).astype(np.float32)
    else:
        from scipy.io import wavfile

        _, wav = wavfile.read(path)
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(1)
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak
    if len(wav) > max_samples:
        start = random.randint(0, len(wav) - max_samples) if train else 0
        wav = wav[start : start + max_samples]
    return torch.from_numpy(wav)


def load_graph(path, max_nodes):
    g = np.load(path)
    x = g["x"].astype(np.float32)[:max_nodes]
    ei = g["edge_index"].astype(np.int64)
    et = g["edge_type"].astype(np.int64)
    keep = (ei[0] < max_nodes) & (ei[1] < max_nodes)
    return torch.from_numpy(x), torch.from_numpy(ei[:, keep]), torch.from_numpy(et[keep])


class MMDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, meta, modalities, train=False, records=None, drop_at_test=()):
        self.root = root
        self.meta = meta
        self.modalities = [m for m in modalities if m in meta["modalities"]]
        self.train = train
        self.records = records if records is not None else read_jsonl(os.path.join(root, f"{split}.jsonl"))
        self.drop = set(drop_at_test)
        self.rotate360 = meta.get("rotate360", False)

    def __len__(self):
        return len(self.records)

    def _path(self, p):
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    def __getitem__(self, i):
        r = self.records[i]
        out = {"id": r["id"], "label": r.get("label", -1), "entity": r.get("entity", -1), "n_choices": r.get("n_choices", -1)}
        for m in self.modalities:
            v = r.get(m)
            if v is None or m in self.drop:
                out[m] = None
            elif m == "text":
                out[m] = v
            elif m == "image":
                out[m] = load_image(self._path(v), self.meta["image_size"], self.train, self.rotate360)
            elif m == "audio":
                out[m] = load_audio(self._path(v), self.meta["max_audio_tokens"] * AUDIO_HOP, self.train)
            elif m == "graph":
                out[m] = load_graph(self._path(v), self.meta["max_graph_nodes"])
        return out


class Collator:
    def __init__(self, meta, modalities, tokenizer):
        self.meta = meta
        self.modalities = [m for m in modalities if m in meta["modalities"]]
        self.tok = tokenizer

    def __call__(self, items):
        B = len(items)
        batch = {
            "ids": [it["id"] for it in items],
            "label": torch.tensor([it["label"] for it in items], dtype=torch.long),
            "entity": torch.tensor([it["entity"] for it in items], dtype=torch.long),
            "present": {},
        }
        if any(it.get("n_choices", -1) > 0 for it in items):
            batch["n_choices"] = torch.tensor([it.get("n_choices", -1) for it in items], dtype=torch.long)
        for m in self.modalities:
            pres = torch.tensor([it[m] is not None for it in items])
            batch["present"][m] = pres
            if m == "text":
                texts = [it[m] if it[m] is not None else "" for it in items]
                batch["text"] = self.tok(texts, self.meta["max_text_len"])
            elif m == "image":
                s = self.meta["image_size"]
                batch["image"] = torch.stack([it[m] if it[m] is not None else torch.zeros(3, s, s) for it in items])
            elif m == "audio":
                L = max([len(it[m]) for it in items if it[m] is not None] + [AUDIO_HOP * 4])
                L = int(np.ceil(L / AUDIO_HOP) * AUDIO_HOP)
                wav = torch.zeros(B, L)
                lens = torch.zeros(B, dtype=torch.long)
                for b, it in enumerate(items):
                    if it[m] is not None:
                        wav[b, : len(it[m])] = it[m]
                        lens[b] = len(it[m])
                batch["audio"] = {"wav": wav, "lengths": lens}
            elif m == "graph":
                F = self.meta["graph_in_dim"]
                N = max([it[m][0].shape[0] for it in items if it[m] is not None] + [1])
                x = torch.zeros(B, N, F)
                nmask = torch.zeros(B, N, dtype=torch.bool)
                eis, ets = [], []
                for b, it in enumerate(items):
                    if it[m] is None:
                        continue
                    gx, ei, et = it[m]
                    n = gx.shape[0]
                    x[b, :n] = gx
                    nmask[b, :n] = True
                    eis.append(ei + b * N)  # flat node index b*N + j
                    ets.append(et)
                ei = torch.cat(eis, 1) if eis else torch.zeros(2, 0, dtype=torch.long)
                et = torch.cat(ets) if ets else torch.zeros(0, dtype=torch.long)
                batch["graph"] = {"x": x, "mask": nmask, "edge_index": ei, "edge_type": et}
        return batch


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch
