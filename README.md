# MMRL: Multi-Modal Scientific Representation Learning

Code for "Multi-Modal Scientific Representation Learning: A Hierarchical
Cross-Modal Framework for Text, Image, and Graph Data" (J. Vijayakumar et al.).

* `mmrl/`: MMRL (MAFE, HCMF, SKGA) and the five baselines
* `train.py`, `profile_cost.py`, `stats.py`, `make_tables.py`: training, cost, statistics, tables and figures
* `tools/`: dataset preparation for ScienceQA, ogbn-arxiv and ROCOv2, entity embeddings, checks
* `kaggle/`, `notebooks/mmrl_kaggle_run.ipynb`: the exact run used for the paper (Kaggle, 2x T4)
* `docs/`: dataset cards, all hyperparameters, the list of every run, measured parameters and FLOPs

The datasets are not redistributed; the scripts fetch them from their original sources under their own licences (see `docs/dataset_cards.md`).

# MMRL experiment pipeline

Code to produce every number in Section V of the MMRL manuscript from real
training runs: Table IV (main comparison), Table V (ablation, including
w/o MAFE and w/o SKGA), Table VI (cost), the convergence, ROC, confusion and
retrieval figures, the robustness and transfer checks in Section VI, and the
significance tests in Section V-H.

Nothing in this folder contains results. The synthetic data generator exists
only to check that the code runs; its numbers must not go into the paper.

**Running on Kaggle or Colab with public datasets:** follow `kaggle/KAGGLE.md`.
It prepares ScienceQA, ogbn-arxiv and ROCOv2 with `tools/prepare_*.py` and
runs the grid in 12-hour sessions with `kaggle/session.sh`.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # PyTorch 2.1+ with CUDA, transformers, scikit-learn, scipy, matplotlib, pillow
```

The `paper` preset downloads SciBERT (`allenai/scibert_scivocab_uncased`) and
ViT-B/16 (`google/vit-base-patch16-224-in21k`) from the Hugging Face hub on
first use.

## 2. Check the code runs (CPU, about 15 minutes)

```bash
python tools/make_synthetic.py --out data/synth_classification --task classification
python tools/make_synthetic.py --out data/synth_vqa --task vqa
python tools/make_synthetic.py --out data/synth_retrieval --task retrieval
python train.py --data data/synth_classification --method mmrl --preset tiny \
    --epochs 2 --bs 16 --accum 1 --workers 0 --out runs/smoke --robustness
```

## 3. Data format

One directory per dataset:

```
data/pubmed10k/
  meta.json
  train.jsonl  val.jsonl  test.jsonl
  corpus.jsonl            (retrieval only)
  images/ audio/ graphs/  (any layout; paths in the jsonl are relative to this dir)
  kg/entity_emb.npy       [num_entities, d_e] float32, KG entity embeddings for SKGA
```

`meta.json`

```json
{"name": "pubmed10k", "task": "classification",
 "modalities": ["text", "image", "graph"],
 "num_classes": 6, "label_names": ["Biology", "..."],
 "graph_in_dim": 64, "num_relations": 12,
 "entity_emb": "kg/entity_emb.npy",
 "max_text_len": 512, "max_audio_tokens": 300, "max_graph_nodes": 128,
 "rotate360": false}
```

`task` is `classification`, `vqa` (add `"answers": [...]`, the normalised
answer vocabulary; `label` is the answer index, so accuracy = exact match) or
`retrieval`.

One jsonl line per sample. Any modality may be missing or `null`; the models
handle absent modalities per sample.

```json
{"id": "PMC1234567", "label": 2, "entity": 815,
 "text": "title. abstract ...",
 "image": "images/PMC1234567_f1.png",
 "audio": "audio/PMC1234567.wav",
 "graph": "graphs/PMC1234567.npz"}
```

* `image`: any RGB file; resized to 224x224, normalised with mean = std = 0.5 (ViT-in21k).
  Set `rotate360: true` in meta.json for molecule drawings.
* `audio`: 16 kHz mono `.wav` or a float32 `.npy` waveform. Cropped to
  `max_audio_tokens x 128` samples (300 tokens = 2.4 s); increase
  `max_audio_tokens` for full narrations (cost grows with it).
* `graph`: `.npz` with `x` [N, graph_in_dim] float, `edge_index` [2, E] int
  (source row, target row), `edge_type` [E] int in `[0, num_relations)`.
  Add reverse edges yourself if you want message passing in both directions.
* `entity`: row index into `entity_emb.npy` of the KG entity the sample is
  about, or -1. Samples with -1 skip the SKGA loss.
* Retrieval: `train/val/test.jsonl` hold queries with
  `"positives": ["doc_id", ...]`; `corpus.jsonl` holds the documents (same
  sample format). Training uses in-batch InfoNCE; evaluation ranks the whole
  corpus and reports mAP@K and Recall@K for K in {1, 5, 10, 20, 50, 100}.

Run `python tools/check_dataset.py data/<name>` on each dataset. It checks
missing files, label ranges, class balance and train/val/test id overlap.

### Where the four datasets can come from

As far as I could find, PubMed-10K, ArXiv-MML, Scientific VQA and BioMed
Retrieval are not published benchmarks under those names, so Section IV-A
must describe exactly how you built each one. Possible sources:

* **PubMed-10K**: PMC Open Access Subset (text, figures, licence per article)
  plus MeSH descriptors from the NLM MeSH files as the graph and KG. A real
  limitation: PMC does not cover Physics or Engineering well, so check that
  the six disciplines you name actually have enough articles.
* **ArXiv-MML**: arXiv metadata plus source or PDF pages for figures, and
  the citation graph from Semantic Scholar or OpenAlex. Audio narrations do
  not exist for arXiv in general. You either generate them with a TTS model
  (say so in the paper) or use an existing set such as `taesiri/arxiv_audio`
  on Hugging Face (check its contents and licence first).
* **Scientific VQA**: ScienceQA (image subset), ChartQA or SciCap-based QA
  are public options. Name the one you use.
* **BioMed Retrieval**: ROCO or MedICaT (figure–caption pairs) can give
  text↔image retrieval. Molecule queries need a source such as PubChem.

Whichever you choose, store the official split if one exists, otherwise a
fixed-seed 70/10/20 split, and record it.

## 4. Run everything

```bash
DATA=data OUT=results bash run_all.sh
```

`run_all.sh` runs, for 5 seeds each:

| What | Command pattern |
|---|---|
| Table IV | `--method {mmrl,cma,mmbert,late,early}` and `--method unimodal --unimodal <m>` per dataset; the UniModal row is the best modality by validation score |
| Robustness (VI-C) | `--robustness` drops each modality at test time only |
| Table V | `--modalities` without one modality (w/o X encoder), `--ablate fusion`, `--ablate attention`, `--ablate mafe` (one shared projection after zero-padding), `--lambda_align 0` (w/o SKGA) |
| Table VI | `profile_cost.py`: batch 32, one GPU, eval mode, synthetic full-length batch, 10 warm-up + 100 timed batches, data loading excluded; peak memory from `torch.cuda.max_memory_allocated`; FLOPs per sample from `torch.utils.flop_counter` |
| V-H | `stats.py`: paired t-test over seeds with Bonferroni, Cohen's d, McNemar per seed |

Then:

```bash
python make_tables.py --results results --ablation_dataset pubmed10k \
   --roc_dataset arxiv_mml --retrieval_dataset biomed_retrieval --out paper_out
```

writes `paper_out/tables.md`, `paper_out/summary.json` and figures 2 to 8.

```bash
python tools/draft_section_v.py paper_out/summary.json > paper_out/section_v.md
```

drafts Section V text whose wording follows the numbers (it says so when a
baseline wins or an ablation does not hurt). Send `summary.json`, `tables.md`
and `results/stats.json` back and Section V can be rewritten from them.

Rough compute: one MMRL run on 10k samples with full-length text (512
tokens) and ViT-B is in the order of 1 to 3 GPU-hours on an A100 with
`--amp`, so the full grid is a few hundred GPU-hours. Reduce with fewer
seeds for baselines first, or run datasets in parallel on several GPUs.

## 5. Settings that the manuscript must state

These are the defaults. If you change one, change the paper.

| Setting | Default |
|---|---|
| Optimiser | AdamW, weight decay 0.01 |
| Learning rates | 2e-5 for the pretrained encoders, 1e-4 for MAFE/fusion/heads (`--lr`, `--head_lr`) |
| Schedule | linear warm-up over 10% of steps, cosine decay |
| Batch | 32 per step x 4 accumulation = 128 |
| Epochs | up to 50, early stopping on validation score, patience 5 |
| Shared width D | 512 |
| SKGA | λ = 0.1, τ = 0.07, contrast against all entities |
| Retrieval loss | symmetric InfoNCE, τ = 0.05, in-batch negatives |
| HCMF | 8 heads; 16 entity units per modality; 1 entity-level layer |
| CMA baseline | MulT-style, 2 cross-attention layers per ordered modality pair, width 768 |
| MM-BERT baseline | MAG gate after BERT layer 1, β = 0.5 |
| Precision | `--amp bf16` on A100/H100, `--amp fp16` (with loss scaling) on T4/P100/V100 |

## 6. Things to know before writing up

* Table IV in the current draft labels every column "Accuracy". Retrieval has
  no accuracy; the pipeline reports mAP@10 as the headline retrieval number.
* The draft's Table III says PubMed-10K has no audio, yet the draft's ablation
  removes the audio encoder on PubMed-10K. The real ablation can only remove
  modalities the dataset has.
* With SciBERT (110M) and ViT-B (86M) as encoders, no model in Table VI can
  have fewer than about 196M parameters, so the draft's 124M for MMRL cannot be
  right. `profile_cost.py` gives the real counts.
* Five seeds give a paired t-test with 4 degrees of freedom; report the
  p-values you get, and use the McNemar results as the per-example check.

## 7. Parameters and FLOPs at paper size (measured, no training needed)

`--preset paper_random` builds the same architectures with random weights, so
parameter counts and FLOPs can be checked without a GPU. With 4 modalities,
12 classes, 512 text tokens, 197 image tokens, 300 audio tokens, 128 graph
nodes (graph_in_dim 128, 4 relations), one sample:

| Method | Params (M) | GFLOPs / sample |
|---|---|---|
| UniModal (text) | 109.7 | 87.0 |
| Early Fusion | 234.6 | 147.6 |
| Late Fusion | 232.9 | 147.6 |
| MM-BERT | 236.2 | 151.8 |
| Cross-Modal Attn. | 440.6 | 249.4 |
| MMRL | 252.0 | 156.9 |

FLOPs count a multiply-add as 2 (torch.utils.flop_counter). Rerun
`profile_cost.py` on your real meta.json (class count, graph feature size and
lengths change these slightly) and on the GPU for latency and memory.
