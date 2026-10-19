# MMRL: Code, Data Preparation and Supplementary Material

Supplementary material for "Multi-Modal Scientific Representation Learning: A
Hierarchical Cross-Modal Framework for Text, Image, and Graph Data".

This repository is the supplementary material of the paper. The code is in `S1_Code/mmrl`; run it from that folder.

## What this package contains

| Folder | Contents | Status |
|---|---|---|
| S1_Code | Complete source: MMRL, the five baselines, training, evaluation, cost profiling, statistics, table and figure generation, dataset preparation, Kaggle session script | Final |
| S2_Datasets | Dataset cards (source, citation, licence, splits, preprocessing) and format examples | Final |
| S3_Experimental_Setup | All hyperparameters (`hyperparameters.json`), the planned runs with exact commands (`run_manifest.csv`), the 75 completed runs with their scores (`runs_completed.csv`), compute plan | Final |
| S4_Measured_Model_Cost | Parameter and FLOP counts for every model, measured with PyTorch's FLOP counter | Final, measured |
| S5_Software_Verification | Logs of end-to-end runs on synthetic and mock data that show every component executes | Final, software tests only |
| S6_Experimental_Results | Results of the 75 GPU training runs on Kaggle (2 x Tesla T4): exported run data, statistics, tables and figures 2 to 8 as in the paper | Final, measured |
| SHA256SUMS.txt | Checksums of every file | Final |

## How to read S5

The logs in S5 come from small randomly initialised encoders trained for one
to twelve epochs on synthetic or mock data on a CPU. They show that the code
runs end to end: data conversion, every model, every ablation, cost
measurement, statistics, tables and figures. Their accuracy numbers carry no
information about MMRL and are not reported in the paper.

## Reproducing the paper

1. Prepare the datasets: `S1_Code/mmrl/kaggle/session.sh` with `STAGE=prep` (see `S1_Code/mmrl/kaggle/KAGGLE.md`).
2. Run the commands in `S3_Experimental_Setup/run_manifest.csv` (or the Kaggle stages).
3. `STAGE=report` writes the tables, figures and statistics that go into S6.

The paper's runs used a reduced budget to fit free Kaggle GPUs: `EPOCHS=4 PATIENCE=2` instead of the 10 and 3 written in `run_manifest.csv`. The exact notebooks are `S1_Code/mmrl/notebooks/mmrl_kaggle_run.ipynb` (run 1) and `mmrl_kaggle_run2.ipynb` (run 2). Code: https://github.com/vjethose/MMRL

The datasets themselves are not redistributed here. All three are public and
are fetched from their original sources by the preparation scripts, which keeps
their licences intact (see S2).
