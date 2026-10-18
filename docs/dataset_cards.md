# Dataset cards

All three datasets are public. They are downloaded from the original sources
and converted by the scripts in `S1_Code/mmrl/tools/`. Subsampling uses seed 0
and is deterministic.

## D1. ScienceQA, image questions

| Field | Value |
|---|---|
| Reference | P. Lu et al., "Learn to explain: Multimodal reasoning via thought chains for science question answering," NeurIPS 35, 2022 |
| Source | Hugging Face `derek-thomas/ScienceQA` (parquet, official splits) |
| Licence | CC BY-NC-SA 4.0 (stated in the ScienceQA paper) |
| Full dataset | 21,208 questions, 10,332 with an image; official split 12,726 / 4,241 / 4,241 |
| Used | All questions with an image, in the official train / validation / test splits (exact counts are written to `meta.json` → `sizes`) |
| Modalities | Text (question, optional context, options), image |
| Task | Multiple choice with 2 to 5 options; label = index of the correct option; options beyond a question's own are masked |
| Metric | Accuracy (= exact match) |
| Not used | `lecture` and `solution` fields (the solution states the answer) |
| KG entities (SKGA) | The question's `skill` label; embedded with frozen SciBERT |
| Script | `tools/prepare_scienceqa.py` |

## D2. ogbn-arxiv

| Field | Value |
|---|---|
| Reference | W. Hu et al., "Open graph benchmark: Datasets for machine learning on graphs," NeurIPS 33, 2020 |
| Source | OGB (`ogb` package) and `titleabs.tsv.gz` from snap.stanford.edu/ogb/data/misc/ogbn_arxiv/ |
| Licence | ODC-BY |
| Full dataset | 169,343 papers, 1,166,243 directed citation edges, 40 subject areas; time split (train ≤ 2017, val 2018, test ≥ 2019) |
| Used | 20,000 / 4,000 / 4,000 papers sampled from the official train / val / test sets |
| Modalities | Text (title + abstract), graph (the paper and up to 31 citation neighbours, induced edges, relations cites / cited-by, OGB 128-d node features) |
| Task | 40-class classification |
| Metric | Accuracy, macro F1, macro AUC |
| KG entities | None; SKGA is disabled for this dataset |
| Script | `tools/prepare_ogbn_arxiv.py` |

## D3. ROCOv2

| Field | Value |
|---|---|
| Reference | J. Rückert et al., "ROCOv2: Radiology Objects in COntext Version 2, an updated multimodal image dataset," Scientific Data 11, 2024, doi:10.1038/s41597-024-03496-6 |
| Source | Zenodo (latest version of the ROCOv2 record; captions, UMLS concepts, images) |
| Licence | CC BY-NC 4.0 (Zenodo record); per-image licences in `license_information.csv` |
| Full dataset | 79,789 radiology images from PMC Open Access with captions and UMLS concepts, official train / valid / test splits |
| Used | 10,000 / 2,000 / 2,000 image–caption pairs sampled from the official splits |
| Modalities | Query: caption (text). Document: image |
| Task | Caption → image retrieval; each split is ranked against its own image pool |
| Metric | mAP@K and Recall@K, K ∈ {1, 5, 10, 20, 50, 100} |
| KG entities (SKGA) | First UMLS concept (CUI) of each image; embedded from the concept name with frozen SciBERT |
| Script | `tools/prepare_rocov2.py` |

## Format

`format_examples/` shows the converted format (meta.json and the first lines
of train.jsonl) produced by the scripts on mock files that mimic each
release's layout. The content of those examples is synthetic.
