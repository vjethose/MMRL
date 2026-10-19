# Compute plan

* Hardware: one NVIDIA T4 or P100 (16 GB) per session on Kaggle Notebooks; fp16 mixed precision with loss scaling.
* Sessions: up to 12 hours each; finished runs are skipped on resume (`kaggle/session.sh`).
* Runs: 85 in total (`run_manifest.csv`): 63 for Table IV (3 datasets, 3 seeds), 18 for Table V, 10 cost profiles for Table VI.
* Estimated GPU time: about 40 hours. This is an estimate; the `speed` stage measures the real per-epoch time before the grid starts.
* Report: `STAGE=report` produces `paper_out/summary.json`, `tables.md`, `stats.json`, `section_v.md` and figures 2 to 8.
