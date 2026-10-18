"""MMRL (Section III) and the five baselines (Section IV-B).

Every model returns a dict with
  emb     [B, D]  document embedding (used for retrieval and SKGA)
  logits  [B, C]  class / answer logits (classification and VQA), or None
All models share the same modality encoders, so differences in results come
from fusion, not from the backbones.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import build_encoders, encode_all, masked_mean


class CrossBlock(nn.Module):
    """Pre-norm cross-attention block: Q from one stream, K/V from another."""

    def __init__(self, d, heads, ffn_mult=2, dropout=0.1):
        super().__init__()
        self.nq, self.nkv, self.nf = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv, kv_mask):
        """kv_mask: True = valid. Caller guarantees at least one valid key per row."""
        a, w = self.attn(self.nq(q), self.nkv(kv), self.nkv(kv), key_padding_mask=~kv_mask, need_weights=False)
        q = q + self.drop(a)
        return q + self.drop(self.ffn(self.nf(q)))


def masked_softmax_pool(scores, present):
    scores = scores.masked_fill(~present, float("-inf"))
    return torch.softmax(scores, dim=-1)


# =========================================================================== MMRL
class MMRL(nn.Module):
    """ablate: subset of {"mafe", "fusion", "attention"}; SKGA is switched off
    in training by setting lambda_align = 0 (see train.py)."""

    def __init__(self, meta, modalities, preset, D=512, heads=8, entity_units=16, ablate=()):
        super().__init__()
        self.mods = list(modalities)
        self.ablate = set(ablate)
        self.D = D
        self.encoders = build_encoders(meta, self.mods, preset)
        dims = {m: self.encoders[m].dim for m in self.mods}

        # ---- MAFE (Eq. 1): one projection per modality; ablation = one shared projection
        if "mafe" in self.ablate:
            self.dmax = max(dims.values())
            self.shared_proj = nn.Sequential(nn.Linear(self.dmax, D), nn.LayerNorm(D), nn.GELU())
        else:
            self.proj = nn.ModuleDict({m: nn.Sequential(nn.Linear(dims[m], D), nn.LayerNorm(D), nn.GELU()) for m in self.mods})
        self.mod_emb = nn.ParameterDict({m: nn.Parameter(torch.zeros(D)) for m in self.mods})
        self.null = nn.Parameter(torch.zeros(1, 1, D))  # always-valid key so absent modalities never yield NaN
        nn.init.normal_(self.null, std=0.02)

        if "fusion" not in self.ablate:
            if "attention" in self.ablate:
                self.mix = nn.ModuleDict({m: nn.Linear(D, D) for m in self.mods})
            else:
                # ---- TLF (Eq. 2)
                self.tlf = nn.ModuleDict({m: CrossBlock(D, heads) for m in self.mods})
                # ---- ELF: learnable pooling into entity units, then attention across units
                self.unit_q = nn.ParameterDict({m: nn.Parameter(torch.randn(entity_units, D) * 0.02) for m in self.mods})
                self.pool = nn.ModuleDict({m: CrossBlock(D, heads) for m in self.mods})
                self.elf = nn.TransformerEncoderLayer(D, heads, 2 * D, 0.1, "gelu", batch_first=True, norm_first=True)
            # ---- DLF (Eq. 3)
            self.g = nn.Sequential(nn.Linear(D, D // 4), nn.GELU(), nn.Linear(D // 4, 1))
        self.out_norm = nn.LayerNorm(D)

    def _mafe(self, m, h):
        if "mafe" in self.ablate:
            h = F.pad(h, (0, self.dmax - h.shape[-1]))
            z = self.shared_proj(h)
        else:
            z = self.proj[m](h)
        return z + self.mod_emb[m]

    def _with_null(self, h, mask):
        B = h.shape[0]
        return torch.cat([self.null.expand(B, -1, -1), h], 1), torch.cat([torch.ones(B, 1, dtype=torch.bool, device=h.device), mask], 1)

    def forward(self, batch):
        feats = encode_all(self.encoders, batch)
        Z = {m: (self._mafe(m, h), mask) for m, (h, mask) in feats.items()}
        mods = list(Z)
        present = torch.stack([batch["present"][m] for m in mods], 1)  # [B, M]

        if "fusion" in self.ablate:  # encoders + MAFE + plain average, no cross-modal interaction
            z = torch.stack([masked_mean(*Z[m]) for m in mods], 1)
            w = present.float() / present.float().sum(1, keepdim=True).clamp(min=1)
            d = (w.unsqueeze(-1) * z).sum(1)
            return {"emb": self.out_norm(d), "alpha": self._full(w, mods)}

        if "attention" in self.ablate:  # same hierarchy, attention replaced by mean mixing
            pooled = {m: masked_mean(*Z[m]) for m in mods}
            z_list = []
            for m in mods:
                others = [pooled[n] * batch["present"][n][:, None] for n in mods if n != m]
                ctx = torch.stack(others, 0).mean(0) if others else torch.zeros_like(pooled[m])
                z_list.append(pooled[m] + self.mix[m](ctx))
            z = torch.stack(z_list, 1)
        else:
            # TLF: each modality attends to the tokens of all other modalities
            T = {}
            for m in mods:
                kv = [Z[n] for n in mods if n != m]
                if kv:
                    k = torch.cat([h for h, _ in kv], 1)
                    km = torch.cat([mk for _, mk in kv], 1)
                    k, km = self._with_null(k, km)
                    T[m] = (self.tlf[m](Z[m][0], k, km), Z[m][1])
                else:
                    T[m] = Z[m]
            # ELF: pool tokens into entity units, then let units of all modalities interact
            units, umask = [], []
            for m in mods:
                h, mk = self._with_null(*T[m])
                B = h.shape[0]
                u = self.pool[m](self.unit_q[m].unsqueeze(0).expand(B, -1, -1), h, mk)
                units.append(u)
                umask.append(batch["present"][m][:, None].expand(-1, u.shape[1]))
            U = torch.cat(units, 1)
            Um = torch.cat(umask, 1)
            U = self.elf(U, src_key_padding_mask=~Um)
            K = units[0].shape[1]
            z = torch.stack([U[:, i * K:(i + 1) * K].mean(1) for i in range(len(mods))], 1)  # [B, M, D]

        # DLF: learned modality weights, normalised over present modalities
        alpha = masked_softmax_pool(self.g(z).squeeze(-1), present)
        d = (alpha.unsqueeze(-1) * z).sum(1)
        return {"emb": self.out_norm(d), "alpha": self._full(alpha, mods)}

    def _full(self, alpha, mods):
        """modality weights over self.mods (zeros for modalities absent from the batch)"""
        full = alpha.new_zeros(alpha.shape[0], len(self.mods))
        for i, m in enumerate(mods):
            full[:, self.mods.index(m)] = alpha[:, i]
        return full


# ====================================================================== baselines
class UniModal(nn.Module):
    def __init__(self, meta, modality, preset, D=512):
        super().__init__()
        self.mods = [modality]
        self.encoders = build_encoders(meta, self.mods, preset)
        self.proj = nn.Sequential(nn.Linear(self.encoders[modality].dim, D), nn.LayerNorm(D), nn.GELU())

    def forward(self, batch):
        feats = encode_all(self.encoders, batch)
        (h, mask), = feats.values()
        return {"emb": self.proj(masked_mean(h, mask))}


class EarlyFusion(nn.Module):
    """Concatenate pooled modality features, then a shared MLP [16]."""

    def __init__(self, meta, modalities, preset, D=512, hidden=1024):
        super().__init__()
        self.mods = list(modalities)
        self.encoders = build_encoders(meta, self.mods, preset)
        self.dims = {m: self.encoders[m].dim for m in self.mods}
        self.mlp = nn.Sequential(nn.Linear(sum(self.dims.values()), hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, D), nn.LayerNorm(D))

    def forward(self, batch):
        feats = encode_all(self.encoders, batch)
        B = batch["label"].shape[0]
        dev = batch["label"].device
        parts = [masked_mean(*feats[m]) if m in feats else torch.zeros(B, self.dims[m], device=dev) for m in self.mods]
        return {"emb": self.mlp(torch.cat(parts, -1))}


class LateFusion(nn.Module):
    """Independent per-modality predictors, averaged at the output [10]."""

    def __init__(self, meta, modalities, preset, D=512, num_out=None):
        super().__init__()
        self.mods = list(modalities)
        self.encoders = build_encoders(meta, self.mods, preset)
        self.proj = nn.ModuleDict({m: nn.Sequential(nn.Linear(self.encoders[m].dim, D), nn.LayerNorm(D), nn.GELU()) for m in self.mods})
        self.heads = nn.ModuleDict({m: nn.Linear(D, num_out) for m in self.mods}) if num_out else None

    def forward(self, batch):
        feats = encode_all(self.encoders, batch)
        mods = list(feats)
        z = {m: self.proj[m](masked_mean(*feats[m])) for m in mods}
        w = torch.stack([batch["present"][m] for m in mods], 1).float()
        w = w / w.sum(1, keepdim=True).clamp(min=1)
        emb = sum(w[:, i:i + 1] * F.normalize(z[m], dim=-1) for i, m in enumerate(mods))
        out = {"emb": emb}
        if self.heads is not None:
            out["logits"] = sum(w[:, i:i + 1] * self.heads[m](z[m]) for i, m in enumerate(mods))
        return out


class CrossModalAttention(nn.Module):
    """Full pairwise directional cross-modal attention, following MulT [34]:
    one stack of cross-attention layers per ordered modality pair, on full
    token sequences, at width `width`."""

    def __init__(self, meta, modalities, preset, D=512, width=768, layers=2, heads=8):
        super().__init__()
        self.mods = list(modalities)
        self.encoders = build_encoders(meta, self.mods, preset)
        if preset == "tiny":
            width = 128
        self.width = width
        self.proj = nn.ModuleDict({m: nn.Linear(self.encoders[m].dim, width) for m in self.mods})
        self.null = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pairs = nn.ModuleDict({f"{m}->{n}": nn.ModuleList(CrossBlock(width, heads, ffn_mult=4) for _ in range(layers))
                                    for m in self.mods for n in self.mods if m != n})
        n_src = max(1, len(self.mods) - 1)
        self.merge = nn.ModuleDict({m: nn.Linear(width * n_src, width) for m in self.mods})
        self.self_attn = nn.ModuleDict({m: nn.TransformerEncoderLayer(width, heads, 4 * width, 0.1, "gelu", batch_first=True, norm_first=True) for m in self.mods})
        self.out = nn.Sequential(nn.Linear(width * len(self.mods), D), nn.LayerNorm(D))

    def forward(self, batch):
        feats = encode_all(self.encoders, batch)
        X = {m: (self.proj[m](h), mk) for m, (h, mk) in feats.items()}
        B = batch["label"].shape[0]
        dev = batch["label"].device
        outs = []
        for m in self.mods:
            if m not in X:
                outs.append(torch.zeros(B, self.width, device=dev))
                continue
            q, qm = X[m]
            streams = []
            for n in self.mods:
                if n == m:
                    continue
                if n in X:
                    kv, km = X[n]
                    kv = torch.cat([self.null.expand(B, -1, -1), kv], 1)
                    km = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), km], 1)
                    s = q
                    for blk in self.pairs[f"{m}->{n}"]:
                        s = blk(s, kv, km)
                else:
                    s = torch.zeros_like(q)
                streams.append(s)
            h = self.merge[m](torch.cat(streams, -1)) if streams else q
            safe = qm.clone()
            safe[:, 0] = True  # keep at least one key per row
            h = self.self_attn[m](h, src_key_padding_mask=~safe)
            outs.append(masked_mean(h, qm) * batch["present"][m][:, None])
        return {"emb": self.out(torch.cat(outs, -1))}


class MMBERT(nn.Module):
    """BERT with a Multimodal Adaptation Gate (MAG-BERT [40]): pooled image,
    audio and graph vectors shift the text hidden states after one layer."""

    def __init__(self, meta, modalities, preset, D=512, inject_layer=1, beta=0.5):
        super().__init__()
        assert "text" in modalities, "MM-BERT needs the text modality"
        self.mods = list(modalities)
        self.encoders = build_encoders(meta, self.mods, preset)
        dt = self.encoders["text"].dim
        self.others = [m for m in self.mods if m != "text"]
        self.gate = nn.ModuleDict({m: nn.Linear(dt + self.encoders[m].dim, dt) for m in self.others})
        self.shift = nn.ModuleDict({m: nn.Linear(self.encoders[m].dim, dt) for m in self.others})
        self.ln = nn.LayerNorm(dt)
        self.beta = beta
        layers = self.encoders["text"].model.encoder.layer
        self.inject_at = min(inject_layer, len(layers) - 1)
        layers[self.inject_at].register_forward_hook(self._hook)
        self._ctx = None
        self.out = nn.Sequential(nn.Linear(dt, D), nn.LayerNorm(D), nn.GELU())

    def _hook(self, module, inp, output):
        if self._ctx is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        H = torch.zeros_like(h)
        for m, (v, pres) in self._ctx.items():
            vv = v[:, None].expand(-1, h.shape[1], -1)
            g = torch.relu(self.gate[m](torch.cat([h, vv], -1)))
            H = H + pres[:, None, None] * g * self.shift[m](vv)
        alpha = torch.clamp(h.norm(dim=-1, keepdim=True) / (H.norm(dim=-1, keepdim=True) + 1e-6) * self.beta, max=1.0)
        h = self.ln(h + alpha * H)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def forward(self, batch):
        ctx = {}
        for m in self.others:
            if m in batch and bool(batch["present"][m].any()):
                h, mk = self.encoders[m](batch[m])
                ctx[m] = (masked_mean(h, mk & batch["present"][m][:, None]), batch["present"][m].float())
        self._ctx = ctx
        try:
            h, mk = self.encoders["text"](batch["text"])
        finally:
            self._ctx = None
        return {"emb": self.out(masked_mean(h, mk))}


# ===================================================================== wrapper
METHODS = ("mmrl", "unimodal", "early", "late", "cma", "mmbert")


class TaskModel(nn.Module):
    """Adds the task head and the SKGA alignment head to a fusion model."""

    def __init__(self, method, meta, modalities, preset, D=512, ablate=(), unimodal=None, entity_emb=None, tau_align=0.07):
        super().__init__()
        self.task = meta["task"]
        n_out = meta.get("num_classes") or len(meta.get("answers", [])) or None
        if self.task == "retrieval":
            n_out = None
        if method == "mmrl":
            self.core = MMRL(meta, modalities, preset, D=D, ablate=ablate)
        elif method == "unimodal":
            self.core = UniModal(meta, unimodal, preset, D=D)
        elif method == "early":
            self.core = EarlyFusion(meta, modalities, preset, D=D)
        elif method == "late":
            self.core = LateFusion(meta, modalities, preset, D=D, num_out=n_out)
        elif method == "cma":
            self.core = CrossModalAttention(meta, modalities, preset, D=D)
        elif method == "mmbert":
            self.core = MMBERT(meta, modalities, preset, D=D)
        else:
            raise ValueError(method)
        self.head = nn.Linear(D, n_out) if (n_out and method != "late") else None
        # SKGA (Eq. 4): contrast the document embedding against all KG entities
        if entity_emb is not None:
            self.register_buffer("entities", torch.as_tensor(entity_emb, dtype=torch.float32))
            self.ent_proj = nn.Linear(self.entities.shape[1], D)
            self.tau_align = tau_align
        else:
            self.entities = None

    def encoder_params(self):
        return [p for n, p in self.named_parameters() if ".encoders." in f".{n}"]

    def forward(self, batch):
        out = self.core(batch)
        if self.head is not None:
            out["logits"] = self.head(out["emb"])
        return out

    def align_loss(self, emb, entity):
        if self.entities is None:
            return emb.new_zeros(())
        keep = entity >= 0
        if not bool(keep.any()):
            return emb.new_zeros(())
        d = F.normalize(emb[keep], dim=-1)
        e = F.normalize(self.ent_proj(self.entities), dim=-1)
        return F.cross_entropy(d @ e.T / self.tau_align, entity[keep])

    def align_score(self, emb, entity):
        """cos(d, e) for the true entity, used for the post-hoc quartile analysis."""
        if self.entities is None:
            return None
        d = F.normalize(emb, dim=-1)
        e = F.normalize(self.ent_proj(self.entities), dim=-1)
        s = (d * e[entity.clamp(min=0)]).sum(-1)
        return torch.where(entity >= 0, s, torch.full_like(s, float("nan")))
