# Running the MMRL experiments on Kaggle's free GPU

The plan uses three public datasets with official splits, so the paper can
cite them and reviewers can reproduce the results:

| Role in paper | Dataset | Modalities | Task | Size used |
|---|---|---|---|---|
| Science VQA | ScienceQA, image questions (Lu et al., NeurIPS 2022) | text + image | 2 to 5-way multiple choice (EM) | full IMG subset, official splits |
| Text + graph classification | ogbn-arxiv (Hu et al., NeurIPS 2020) | text + citation graph | 40 arXiv CS categories | 20k train / 4k val / 4k test from the official time split |
| Cross-modal retrieval | ROCOv2 (Rückert et al., Sci. Data 2024) | caption → radiology image | mAP@K | 10k train / 2k val / 2k test from the official splits |

None of these has audio. The audio encoder stays in the code, but the paper
must say that audio was not evaluated (see the note at the end).

## One-time setup

1. Kaggle account with phone verification (needed for GPU and internet).
2. New notebook → Settings: Accelerator **GPU T4 x2** or **P100**, Internet **On**.
3. Upload `MMRL_pipeline.zip` as a Kaggle Dataset (e.g. `mmrl-code`) and add it to the notebook.
4. ROCOv2: download the release from Zenodo (record 10821435), check the file
   sizes, and upload the captions/concepts CSVs, `cui_mapping.csv` and the image
   folders as a Kaggle Dataset (e.g. `rocov2`). Add it to the notebook.

## Each session

First cell (adapt the paths to your dataset names):

```bash
!cp -r /kaggle/input/mmrl-code/mmrl /kaggle/working/mmrl
%cd /kaggle/working/mmrl
!pip install -q -r requirements.txt
```

Second cell, one stage per session:

```bash
!STAGE=prep ROCO_ROOT=/kaggle/input/rocov2 bash kaggle/session.sh
```

Then **Save Version → Save & Run All (Commit)**. A committed run continues
in the background for up to 12 hours, and everything in `/kaggle/working`
(data and results) is kept as that version's output.

For the next session, add the previous version's output as an input
(Add Input → Your Work → the notebook) and pass it as `PREV`:

```bash
!STAGE=speed PREV=/kaggle/input/<your-notebook-name> bash kaggle/session.sh
```

Order of stages:

| # | STAGE | What it does | Rough GPU time |
|---|---|---|---|
| 1 | prep | download, convert, embed entities, check | 1 to 2 h |
| 2 | speed | one short MMRL epoch on 640 ScienceQA samples; read the s/epoch in the log | 5 min |
| 3 | sqa | 3 seeds x (5 methods + 2 unimodal) | 8 to 12 h |
| 4 | arxiv | 3 seeds x (5 methods + 2 unimodal) | 8 to 12 h |
| 5 | roco | 3 seeds x 5 methods (UniModal does not apply to text-to-image retrieval) | 6 to 10 h |
| 6 | ablation | 3 seeds x 6 ablations on ScienceQA | 6 to 9 h |
| 7 | cost | latency, memory, params, FLOPs on this GPU | 15 min |
| 8 | report | stats, tables, figures, Section V draft → `paper_out.zip` | 2 min |

The times are my estimates, not measurements; stage 2 tells you the real
speed. If a stage stops at the 12-hour limit, run the same stage again with
`PREV` set: finished runs are skipped. Total is roughly 40 GPU hours, which is
more than one week of Kaggle's free quota (Kaggle shows your quota in the
notebook settings), so plan on about two weeks, or run some stages on Colab.

If memory runs out, use `BS=8 ACCUM=4` (same effective batch of 32).

Settings used (state these in Section IV-C): up to 10 epochs, patience 3,
batch 16 x 2 accumulation = 32, fp16 mixed precision, AdamW, learning rate
2e-5 (encoders) and 1e-4 (fusion and heads), 3 seeds.

## What to send back

`paper_out.zip` from stage 8. It contains `summary.json`, `tables.md`,
`stats.json`, `section_v.md` and the figures. With it, Section V, the abstract
and the conclusion can be rewritten from real numbers.

## Paper changes this plan implies

* Section IV-A is rewritten around ScienceQA, ogbn-arxiv and ROCOv2, with
  citations and licences.
* Audio is not evaluated. Either keep the audio encoder in Section III as
  supported but untested and change the title to "Text, Image, and Graph
  Data", or add an audio dataset later.
* Table V moves to ScienceQA (text + image), so the encoder rows become
  "w/o text" and "w/o image".
* SKGA entities are ScienceQA skills and ROCOv2 UMLS concepts, embedded with
  frozen SciBERT; ogbn-arxiv has no KG, so SKGA is off there.
