## Table IV  Main metric (accuracy %; mAP@10 % for retrieval; EM % for VQA)

| Method | ogbn_arxiv | rocov2 | scienceqa |
|---|---|---|---|
| UniModal | 71.4 ± 0.6 | - | 64.2 ± 0.6 |
| Early Fusion | 71.8 ± 0.5 | 8.9 ± 0.4 | 64.5 ± 12.9 |
| Late Fusion | 70.6 ± 0.5 | 7.3 ± 0.3 | 77.4 ± 2.4 |
| MM-BERT | 71.5 ± 0.4 | 2.3 ± 0.2 | 68.7 ± 5.2 |
| Cross-Modal Attn. | 71.6 ± 0.2 | 0.1 ± 0.0 | 73.2 ± 0.4 |
| MMRL (Proposed) | 71.5 ± 0.5 | 7.5 ± 0.4 | 77.4 ± 1.0 |
| Gain (pts) | -0.4 vs Early Fusion | -1.5 vs Early Fusion | +0.0 vs Late Fusion |

UniModal row uses the best single modality by validation score: {'ogbn_arxiv': 'text', 'scienceqa': 'text'}

## Table V  Ablation on scienceqa

| Configuration | Accuracy | F1 | mAP | Δ Acc |
|---|---|---|---|---|
| MMRL (Full) | 77.4 ± 1.0 | nan ± nan | nan ± nan | - |
| w/o Text Encoder | 71.8 ± 3.6 | nan ± nan | nan ± nan | -5.6 |
| w/o Image Encoder | 67.2 ± 1.4 | nan ± nan | nan ± nan | -10.2 |
| w/o Cross-Modal Fusion | 77.4 ± 1.7 | nan ± nan | nan ± nan | +0.0 |
| w/o Attention Mechanism | 78.2 ± 0.6 | nan ± nan | nan ± nan | +0.8 |
| w/o MAFE (shared projection) | 71.4 ± 11.1 | nan ± nan | nan ± nan | -6.0 |
| w/o SKGA (λ = 0) | 76.6 ± 1.2 | nan ± nan | nan ± nan | -0.8 |

## Table VI  Cost (batch 32, single GPU, data loading excluded)

| Method | Latency ms/batch | ms/sample | Peak GPU MB | Params M | GFLOPs/sample | Dataset |
|---|---|---|---|---|---|---|
| UniModal | 194.4 | 6.08 | 612 | 110.1 | 73.3 | scienceqa |
| Early Fusion | 291.4 | 9.11 | 964 | 197.6 | 108.4 | scienceqa |
| Late Fusion | 301.2 | 9.41 | 961 | 196.3 | 108.4 | scienceqa |
| MM-BERT | 271.8 | 8.49 | 986 | 197.7 | 109.3 | scienceqa |
| Cross-Modal Attn. | 304.2 | 9.50 | 1189 | 241.2 | 129.7 | scienceqa |
| MMRL (Proposed) | 237.3 | 7.42 | 1014 | 207.3 | 111.6 | scienceqa |

MMRL vs CMA, % reduction: {"latency_ms_per_batch": 22.0, "peak_mem_MB": 14.7, "gflops_per_sample": 13.9, "params_M": 14.1}

## Convergence on scienceqa (epoch of best validation score)

- Early Fusion: best epoch 4.0 ± 0.0, train loss there 0.652 ± 0.266
- Late Fusion: best epoch 4.0 ± 0.0, train loss there 0.353 ± 0.104
- MM-BERT: best epoch 4.0 ± 0.0, train loss there 0.574 ± 0.115
- Cross-Modal Attn.: best epoch 4.0 ± 0.0, train loss there 0.443 ± 0.012
- MMRL (Proposed): best epoch 3.7 ± 0.6, train loss there 0.396 ± 0.089

## Macro AUC on ogbn_arxiv: mmrl 0.969 ± 0.006, unimodal_graph 0.907 ± 0.002, unimodal_text 0.970 ± 0.002

## mAP@K on rocov2

| Method | K=1 | K=5 | K=10 | K=20 | K=50 | K=100 |
|---|---|---|---|---|---|---|
| Early Fusion | 4.0 | 7.7 | 8.9 | 9.8 | 10.5 | 10.7 |
| Late Fusion | 3.3 | 6.1 | 7.3 | 8.0 | 8.7 | 9.0 |
| MM-BERT | 0.9 | 1.8 | 2.3 | 2.7 | 3.2 | 3.5 |
| Cross-Modal Attn. | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 0.3 |
| MMRL (Proposed) | 3.3 | 6.4 | 7.5 | 8.3 | 9.0 | 9.3 |

## Accuracy with one modality dropped at inference only

- ogbn_arxiv MMRL (Proposed): no text 3.3 ± 0.7, no graph 71.1 ± 0.4
- ogbn_arxiv Cross-Modal Attn.: no text 8.3 ± 11.0, no graph 8.7 ± 11.7
- scienceqa MMRL (Proposed): no text 75.3 ± 1.1, no image 60.7 ± 1.3
- scienceqa Cross-Modal Attn.: no text 50.0 ± 1.0, no image 48.5 ± 1.0
