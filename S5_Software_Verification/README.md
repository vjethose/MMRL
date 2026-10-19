# Software verification (not experimental results)

These logs show that every part of the pipeline executes end to end. They were
produced on a 2-core CPU with small randomly initialised encoders
(`--preset tiny`) on synthetic or mock data, for 1 to 12 epochs. **None of the
numbers in these logs is a result about MMRL, and none appears in the paper.**

| Log | What it checks |
|---|---|
| `logs/log_synth_*`, `logs/log_ab_*`, `logs/log_uni_*` | All six methods and all ablations run on synthetic classification, VQA and retrieval data |
| `logs/learn_*.txt` | MMRL, Early Fusion and Cross-Modal Attention reduce training loss and reach high validation accuracy on an easy synthetic task within 12 epochs, i.e. the training loop learns |
| `logs/mock_*.txt` | Training on the outputs of the three dataset converters, run on mock files laid out like ScienceQA, ogbn-arxiv and ROCOv2 |
| `kaggle_flow_mock_tables.md`, `kaggle_flow_mock_section_v.md` | The full Kaggle flow (main grid, ablation, cost, statistics, tables, figures, Section V draft) on mock data |
