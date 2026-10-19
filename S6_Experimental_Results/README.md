# Experimental results (S6)

Every number in Section V of the paper comes from this folder.

## Where the runs were done

| Session | Kaggle notebook | Hardware | What it produced |
|---|---|---|---|
| Run 1 | vjithoshshiva/notebook2b5b23c132 (Version 1) | 2 x Tesla T4, fp16 | ogbn-arxiv and ROCOv2 main grids, ogbn-arxiv cost |
| Run 2 | vjithoshshiva/notebook9c6f324979 (Version 2, 4.83 h) | 2 x Tesla T4, fp16 | ScienceQA main grid and ablations, ScienceQA cost, final report |

Software: Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 5.16.1.
Protocol: at most 4 epochs, patience 2, batch 16 x 2 accumulation, seeds 0, 1, 2.

Run 1 also trained ScienceQA, but with a scoring head that ignored the text of
the answer options, so every model stayed at chance (about 35.6%, against
35.5% mean chance). Run 2 replaced that head with the option scorer described
in Section IV-A and retrained all ScienceQA runs; the run 1 ScienceQA results
are not used anywhere in the paper.

## Files

* `figure_data/run2_figdata.json`: everything `make_tables.py` reads, exported from the run 2 output (75 training runs, 12 cost measurements, dataset metadata, statistics). SHA-256 `2d3519847b2f73ea2ae1ea0e23006dc01f4ca593b313df073459801a91a04554`.
* `figure_data/run1_figdata.json`: the same export from run 1, kept for transparency. SHA-256 `7e92a14952e902ad5492db895eadd70b5455897cd51fc66b4f28690c4ee5fd19`.
* `paper_out/`: `tables.md` (Tables IV to VI and the other summaries), `summary.json`, `stats.json` (t-tests, Cohen's d, McNemar), figures 2 to 8 exactly as in the paper.

The ogbn-arxiv AUC and ROC curve were computed by the exporter from the saved
per-example test probabilities, because run 1 used an older metric routine
that returned no AUC when a class was absent from the sampled test set.

## Rebuilding the tables and figures

    python S1_Code/mmrl/tools/export_figdata.py --unpack S6_Experimental_Results/figure_data/run2_figdata.json --to rebuilt
    python S1_Code/mmrl/make_tables.py --results rebuilt/results --cost rebuilt/results/cost_scienceqa \
        --ablation_dataset scienceqa --roc_dataset ogbn_arxiv --retrieval_dataset rocov2 \
        --confusion_dataset ogbn_arxiv --out rebuilt/paper_out

## The 75 training runs

| Dataset | Runs |
|---|---|
| ScienceQA | 39 (5 fused methods + 2 single-modality models, plus 6 ablations, each x 3 seeds) |
| ogbn-arxiv | 21 (5 fused methods + 2 single-modality models, x 3 seeds) |
| ROCOv2 | 15 (5 fused methods x 3 seeds; single-modality models do not apply to cross-modal retrieval) |

One line per run, with its test score and best epoch: `S3_Experimental_Setup/runs_completed.csv`.
