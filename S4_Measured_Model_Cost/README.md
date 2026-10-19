# Measured parameters and FLOPs

Measured with `profile_cost.py --preset paper_random`, which builds exactly
the architectures used in the paper (SciBERT-size text encoder, ViT-B/16,
R-GCN, fusion modules) with random weights. Parameter counts and FLOPs do not
depend on the weights, so these numbers are final. Latency and memory depend
on the GPU and are measured in the GPU runs (S6).

* FLOPs: `torch.utils.flop_counter.FlopCounterMode`, one sample, forward pass, multiply-add counted as 2.
* Inputs at full length for each configuration:
  * `scienceqa_config`: 256 text tokens + 197 image tokens, 5 answer options.
  * `ogbn_arxiv_config`: 256 text tokens + 32-node graph (128-d features, 2 relations), 40 classes.
  * `four_modality_config`: 512 text + 197 image + 300 audio tokens + 128-node graph, 12 classes (the full framework, for reference).
* The SKGA entity table is a frozen buffer and is not counted as parameters.

| Configuration | MMRL vs Cross-Modal Attention: params | FLOPs |
|---|---|---|
| ScienceQA (text + image) | 206.9 M vs 240.8 M (−14.1%) | 80.0 G vs 91.1 G (−12.2%) |
| ogbn-arxiv (text + graph) | 121.0 M vs 155.1 M (−22.0%) | 45.3 G vs 52.3 G (−13.4%) |
| Four modalities (reference) | 252.0 M vs 440.6 M (−42.8%) | 156.9 G vs 249.4 G (−37.1%) |

Full per-method numbers are in the CSV files and the raw JSONs in `raw/`.

## Note on Table VI of the paper

Table VI and Fig. 7 use the GPU measurements from the Kaggle runs (one Tesla T4,
fp16, batch 32), stored in `S6_Experimental_Results/figure_data/run2_figdata.json`
(cost section) and printed in `S6_Experimental_Results/paper_out/tables.md`.
The ScienceQA CSV in this folder was measured before answer-option scoring was
added, so its FLOPs are lower (80.0 G for MMRL instead of 111.6 G); the
ogbn-arxiv and four-modality CSVs are unaffected. The four-modality figures
(37.1% fewer FLOPs, 42.8% fewer parameters than Cross-Modal Attention) cited
in Section V-F come from this folder.
