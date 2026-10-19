## Table IV  Main metric (accuracy %; mAP@10 % for retrieval; EM % for VQA)

| Method | ogbn_arxiv | rocov2 | scienceqa |
|---|---|---|---|
| UniModal | 1.0 ± 1.4 | - | 31.2 ± 8.8 |
| Early Fusion | 4.0 ± 2.8 | 24.3 ± 0.1 | 78.1 ± 22.1 |
| Late Fusion | 5.0 ± 4.2 | 29.6 ± 2.4 | 62.5 ± 0.0 |
| MM-BERT | 2.0 ± 2.8 | 33.4 ± 12.3 | 50.0 ± 0.0 |
| Cross-Modal Attn. | 2.0 ± 0.0 | 24.4 ± 0.0 | 71.9 ± 22.1 |
| MMRL (Proposed) | 1.0 ± 1.4 | 24.4 ± 0.0 | 78.1 ± 22.1 |
| Gain (pts) | -4.0 vs Late Fusion | -9.0 vs MM-BERT | +0.0 vs Early Fusion |

UniModal row uses the best single modality by validation score: {'ogbn_arxiv': 'graph', 'scienceqa': 'text'}

## Table V  Ablation on scienceqa

| Configuration | Accuracy | F1 | mAP | Δ Acc |
|---|---|---|---|---|
| MMRL (Full) | 78.1 ± 22.1 | nan ± nan | nan ± nan | - |
| w/o Text Encoder | 96.9 ± 4.4 | nan ± nan | nan ± nan | +18.8 |
| w/o Image Encoder | 40.6 ± 13.3 | nan ± nan | nan ± nan | -37.5 |
| w/o Cross-Modal Fusion | 53.1 ± 22.1 | nan ± nan | nan ± nan | -25.0 |
| w/o Attention Mechanism | 59.4 ± 4.4 | nan ± nan | nan ± nan | -18.8 |
| w/o MAFE (shared projection) | 71.9 ± 4.4 | nan ± nan | nan ± nan | -6.2 |
| w/o SKGA (λ = 0) | 68.8 ± 8.8 | nan ± nan | nan ± nan | -9.4 |

## Table VI  Cost (batch 32, single GPU, data loading excluded)

| Method | Latency ms/batch | ms/sample | Peak GPU MB | Params M | GFLOPs/sample | Dataset |
|---|---|---|---|---|---|---|
| UniModal | 68.4 | 2.14 | nan | 4.4 | 0.2 | scienceqa |
| Early Fusion | 116.2 | 3.63 | nan | 5.7 | 0.4 | scienceqa |
| Late Fusion | 116.3 | 3.64 | nan | 5.0 | 0.4 | scienceqa |
| MM-BERT | 113.8 | 3.56 | nan | 5.0 | 0.4 | scienceqa |
| Cross-Modal Attn. | 475.1 | 14.85 | nan | 6.3 | 0.8 | scienceqa |
| MMRL (Proposed) | 744.5 | 23.26 | nan | 16.0 | 2.9 | scienceqa |

MMRL vs CMA, % reduction: {"latency_ms_per_batch": -56.7, "peak_mem_MB": NaN, "gflops_per_sample": -275.6, "params_M": -155.1}

## Convergence on scienceqa (epoch of best validation score)

- Early Fusion: best epoch 1.0 ± 0.0, train loss there 1.063 ± 0.012
- Late Fusion: best epoch 1.0 ± 0.0, train loss there 1.094 ± 0.014
- MM-BERT: best epoch 1.0 ± 0.0, train loss there 1.112 ± 0.007
- Cross-Modal Attn.: best epoch 1.0 ± 0.0, train loss there 1.085 ± 0.071
- MMRL (Proposed): best epoch 1.0 ± 0.0, train loss there 1.648 ± 0.007

## Macro AUC on ogbn_arxiv: mmrl nan ± nan, unimodal_graph nan ± nan, unimodal_text nan ± nan

## mAP@K on rocov2

| Method | K=1 | K=5 | K=10 | K=20 | K=50 | K=100 |
|---|---|---|---|---|---|---|
| Early Fusion | 8.3 | 19.0 | 24.3 | 25.8 | 25.8 | 25.8 |
| Late Fusion | 12.5 | 25.6 | 29.6 | 31.1 | 31.1 | 31.1 |
| MM-BERT | 20.8 | 26.2 | 33.4 | 34.1 | 34.1 | 34.1 |
| Cross-Modal Attn. | 8.3 | 19.0 | 24.4 | 25.9 | 25.9 | 25.9 |
| MMRL (Proposed) | 8.3 | 19.0 | 24.4 | 25.9 | 25.9 | 25.9 |

## Accuracy with one modality dropped at inference only

- ogbn_arxiv MMRL (Proposed): no text 1.0 ± 1.4, no graph 0.0 ± 0.0
- ogbn_arxiv Cross-Modal Attn.: no text 0.0 ± 0.0, no graph 1.0 ± 1.4
- rocov2 Cross-Modal Attn.: no text 24.4 ± 0.0, no image 24.4 ± 0.0
- scienceqa MMRL (Proposed): no text 68.8 ± 26.5, no image 43.8 ± 8.8
- scienceqa Cross-Modal Attn.: no text 43.8 ± 8.8, no image 46.9 ± 4.4
