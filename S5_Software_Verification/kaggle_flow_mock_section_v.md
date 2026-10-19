## A. Overall Performance Comparison

Table IV compares MMRL with the baselines on all 3 datasets (mean ± SD over 2 seeds). On ogbn_arxiv, Late Fusion scores 4.0 points higher than MMRL. On rocov2, MM-BERT scores 9.0 points higher than MMRL. On scienceqa, Early Fusion scores -0.0 points higher than MMRL. 
- ogbn_arxiv: MMRL 1.0 ± 1.4 vs Late Fusion 5.0 ± 4.2
- rocov2: MMRL 24.4 ± 0.0 vs MM-BERT 33.4 ± 12.3
- scienceqa: MMRL 78.1 ± 22.1 vs Early Fusion 78.1 ± 22.1

The UniModal row uses the best single modality on validation data: {'ogbn_arxiv': 'graph', 'scienceqa': 'text'}.

## B. Convergence Analysis

On scienceqa, Early Fusion reaches its best validation score at epoch 1.0; Late Fusion reaches its best validation score at epoch 1.0; MM-BERT reaches its best validation score at epoch 1.0; Cross-Modal Attention reaches its best validation score at epoch 1.0; MMRL reaches its best validation score at epoch 1.0. Training loss alone does not show better generalization, so we rely on the held-out results in Table IV.

## C. ROC Curve Analysis

Macro-averaged one-vs-rest AUC: MMRL nan, graph-only nan, text-only nan.

## E. Ablation Study

Table V reports the ablation on scienceqa. The full model reaches 78.1% accuracy.
- w/o Image Encoder lowers accuracy by 37.5 points.
- w/o Cross-Modal Fusion lowers accuracy by 25.0 points.
- w/o Attention Mechanism lowers accuracy by 18.8 points.
- w/o SKGA (λ = 0) lowers accuracy by 9.4 points.
- w/o MAFE (shared projection) lowers accuracy by 6.2 points.
- w/o Text Encoder raises accuracy by 18.8 points.

## F. Computational Efficiency

Compared with Cross-Modal Attention, MMRL has 56.7% higher inference latency, 275.6% higher FLOPs per sample, 155.1% higher parameter count (744.5 vs 475.1 ms per batch of 32; nan vs nan MB; 2.9 vs 0.8 GFLOPs; 16.0M vs 6.3M parameters), measured on cpu.

## G. Cross-Modal Retrieval Performance

At K = 1, MMRL reaches 8.3 mAP against 20.8 for MM-BERT; at K = 10, 24.4 vs 33.4; at K = 100, 25.9 vs 34.1.

## H. Statistical Significance

- scienceqa, MMRL vs mmbert: diff +28.12, paired t p = 1 (Bonferroni), Cohen's d = 1.27, McNemar p ≤ 1
- scienceqa, MMRL vs late: diff +15.62, paired t p = 1 (Bonferroni), Cohen's d = 0.71, McNemar p ≤ 1
- scienceqa, MMRL vs unimodal: diff +46.88, paired t p = 1 (Bonferroni), Cohen's d = 1.52, McNemar p ≤ 1
- rocov2, MMRL vs early: diff +0.07, paired t p = 1 (Bonferroni), Cohen's d = 0.71
- rocov2, MMRL vs mmbert: diff -9.01, paired t p = 1 (Bonferroni), Cohen's d = -0.74
- rocov2, MMRL vs late: diff -5.19, paired t p = 0.817 (Bonferroni), Cohen's d = -2.13
- ogbn_arxiv, MMRL vs cma: diff -1.00, paired t p = 1 (Bonferroni), Cohen's d = -0.71, McNemar p ≤ 1
- ogbn_arxiv, MMRL vs late: diff -4.00, paired t p = 1 (Bonferroni), Cohen's d = -0.71, McNemar p ≤ 1
- ogbn_arxiv, MMRL vs mmbert: diff -1.00, paired t p = 1 (Bonferroni), Cohen's d = -0.71, McNemar p ≤ 1
- ogbn_arxiv, MMRL vs early: diff -3.00, paired t p = 1 (Bonferroni), Cohen's d = -2.12, McNemar p ≤ 1

## Numbers for Section VI

- ogbn_arxiv MMRL, one modality dropped at test: no text 1.0, no graph 0.0
- ogbn_arxiv Cross-Modal Attention, one modality dropped at test: no text 0.0, no graph 1.0
- rocov2 Cross-Modal Attention, one modality dropped at test: no text 24.4, no image 24.4
- scienceqa MMRL, one modality dropped at test: no text 68.8, no image 43.8
- scienceqa Cross-Modal Attention, one modality dropped at test: no text 43.8, no image 46.9
- scienceqa SKGA alignment quartiles: top 50.0% vs bottom 100.0% accuracy
