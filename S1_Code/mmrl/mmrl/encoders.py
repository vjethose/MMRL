"""Modality-specific encoders (Section III-B).

Every encoder returns (H, mask): H is [B, N, d_m], mask is [B, N] with True
for valid positions.
"""
import math

import torch
import torch.nn as nn

PRESETS = {
    # the configuration described in the paper
    "paper": {
        "text_model": "allenai/scibert_scivocab_uncased",
        "image_model": "google/vit-base-patch16-224-in21k",
        "audio_width": 512, "audio_layers": 6, "audio_heads": 8,
        "graph_width": 256, "graph_layers": 3,
    },
    # same architecture as "paper" with random weights: for offline profiling of params/FLOPs only
    "paper_random": {
        "text_model": None, "image_model": None, "vocab_size": 31090,
        "text_width": 768, "text_layers": 12, "image_width": 768, "image_layers": 12,
        "audio_width": 512, "audio_layers": 6, "audio_heads": 8,
        "graph_width": 256, "graph_layers": 3,
    },
    # small random-init encoders for CPU smoke tests (no downloads)
    "tiny": {
        "text_model": None, "image_model": None,
        "text_width": 128, "text_layers": 2, "image_width": 128, "image_layers": 2,
        "audio_width": 64, "audio_layers": 2, "audio_heads": 4,
        "graph_width": 64, "graph_layers": 3,
    },
}


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import BertConfig, BertModel, AutoModel

        if cfg["text_model"]:
            self.model = AutoModel.from_pretrained(cfg["text_model"], add_pooling_layer=False)
        else:
            w = cfg["text_width"]
            self.model = BertModel(BertConfig(hidden_size=w, num_hidden_layers=cfg["text_layers"],
                                              num_attention_heads=max(1, w // 64), intermediate_size=4 * w,
                                              max_position_embeddings=512, vocab_size=cfg.get("vocab_size", 30522)),
                                   add_pooling_layer=False)
        self.dim = self.model.config.hidden_size

    def forward(self, text):
        out = self.model(input_ids=text["input_ids"], attention_mask=text["attention_mask"])
        return out.last_hidden_state, text["attention_mask"].bool()


class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import ViTConfig, ViTModel

        if cfg["image_model"]:
            self.model = ViTModel.from_pretrained(cfg["image_model"], add_pooling_layer=False)
        else:
            w = cfg["image_width"]
            self.model = ViTModel(ViTConfig(hidden_size=w, num_hidden_layers=cfg["image_layers"],
                                            num_attention_heads=max(1, w // 64), intermediate_size=4 * w,
                                            image_size=224, patch_size=16), add_pooling_layer=False)
        self.dim = self.model.config.hidden_size

    def forward(self, pixels):
        h = self.model(pixel_values=pixels).last_hidden_state  # [B, 197, d]
        return h, torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)


class AudioEncoder(nn.Module):
    """1D conv front end (kernels 64/32/16/8, strides 8/4/2/2) + Transformer."""

    def __init__(self, cfg, max_tokens=300):
        super().__init__()
        w = cfg["audio_width"]
        layers, cin = [], 1
        for k, s in zip((64, 32, 16, 8), (8, 4, 2, 2)):
            layers += [nn.Conv1d(cin, w, k, s, padding=(k - s) // 2), nn.GroupNorm(1, w), nn.GELU()]
            cin = w
        self.conv = nn.Sequential(*layers)
        self.hop = 8 * 4 * 2 * 2
        self.pos = nn.Parameter(torch.zeros(1, 4096, w))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(w, cfg["audio_heads"], 4 * w, 0.1, "gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, cfg["audio_layers"], enable_nested_tensor=False)
        self.dim = w

    def forward(self, audio):
        h = self.conv(audio["wav"].unsqueeze(1)).transpose(1, 2)  # [B, T, w]
        T = h.shape[1]
        mask = torch.arange(T, device=h.device)[None] < (audio["lengths"] // self.hop).clamp(min=1)[:, None]
        h = h + self.pos[:, :T]
        h = self.tf(h, src_key_padding_mask=~mask)
        return h, mask


class RGCNLayer(nn.Module):
    """Relational GCN layer [27] with basis decomposition and mean aggregation."""

    def __init__(self, din, dout, num_rel, num_bases=8):
        super().__init__()
        nb = min(num_bases, num_rel)
        self.bases = nn.Parameter(torch.empty(nb, din, dout))
        self.coef = nn.Parameter(torch.empty(num_rel, nb))
        self.self_loop = nn.Linear(din, dout)
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coef)

    def forward(self, x, edge_index, edge_type):
        out = self.self_loop(x)
        if edge_index.numel() == 0:
            return out
        src, dst = edge_index
        xb = torch.einsum("ni,bio->nbo", x, self.bases)  # [N, nb, dout]
        msg = torch.einsum("eb,ebo->eo", self.coef[edge_type], xb[src])  # [E, dout]
        agg = torch.zeros_like(out).index_add_(0, dst, msg)
        deg = torch.zeros(x.shape[0], device=x.device).index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        return out + agg / deg.clamp(min=1)[:, None]


class GraphEncoder(nn.Module):
    def __init__(self, cfg, in_dim, num_rel):
        super().__init__()
        w = cfg["graph_width"]
        dims = [in_dim] + [w] * cfg["graph_layers"]
        self.layers = nn.ModuleList(RGCNLayer(a, b, max(1, num_rel)) for a, b in zip(dims[:-1], dims[1:]))
        self.drop = nn.Dropout(0.1)
        self.dim = w

    def forward(self, g):
        B, N, F = g["x"].shape
        h = g["x"].reshape(B * N, F)
        for i, layer in enumerate(self.layers):
            h = layer(h, g["edge_index"], g["edge_type"])
            if i < len(self.layers) - 1:
                h = self.drop(torch.relu(h))
        return h.reshape(B, N, -1), g["mask"]


def build_encoders(meta, modalities, preset):
    cfg = PRESETS[preset]
    enc = nn.ModuleDict()
    for m in modalities:
        if m == "text":
            enc[m] = TextEncoder(cfg)
        elif m == "image":
            enc[m] = ImageEncoder(cfg)
        elif m == "audio":
            enc[m] = AudioEncoder(cfg, meta["max_audio_tokens"])
        elif m == "graph":
            enc[m] = GraphEncoder(cfg, meta["graph_in_dim"], meta.get("num_relations", 1))
    return enc


def encode_all(encoders, batch):
    """Run every encoder; zero the mask of samples where the modality is absent."""
    feats = {}
    for m, enc in encoders.items():
        if m not in batch:
            continue
        pres = batch["present"][m]
        if not bool(pres.any()):
            continue
        h, mask = enc(batch[m])
        feats[m] = (h, mask & pres[:, None])
    return feats


def masked_mean(h, mask):
    w = mask.unsqueeze(-1).to(h.dtype)
    return (h * w).sum(1) / w.sum(1).clamp(min=1.0)
