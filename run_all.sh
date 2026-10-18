#!/usr/bin/env bash
# Every run needed for Section V. Edit the dataset names and paths first.
# Roughly: 4 datasets x (5 baselines + up to 4 unimodal + MMRL) x 5 seeds,
# plus 8 ablations x 5 seeds on the ablation dataset.
set -euo pipefail

DATA=${DATA:-data}
OUT=${OUT:-results}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DATASETS=${DATASETS:-"pubmed10k arxiv_mml sci_vqa biomed_retrieval"}
ABL=${ABL:-pubmed10k}          # dataset for Table V
ROC=${ROC:-arxiv_mml}          # dataset for the ROC figure
COMMON=${COMMON:-"--amp ${AMP:-bf16} --workers 8 --out $OUT"}

mods_of() { python -c "import json,sys;print(' '.join(json.load(open('$DATA/$1/meta.json'))['modalities']))"; }
without() { python -c "import sys;m=sys.argv[1].split();print(','.join(x for x in m if x!=sys.argv[2]))" "$1" "$2"; }
done_already() { [ -f "$OUT/$1__s$2.json" ]; }

for ds in $DATASETS; do
  python tools/check_dataset.py "$DATA/$ds"
done

# ---- Table IV (main comparison) + robustness + ROC inputs
for seed in $SEEDS; do
  for ds in $DATASETS; do
    M=$(mods_of $ds)
    for method in mmrl cma mmbert late early; do
      done_already "${ds}__${method}" $seed || \
        python train.py --data $DATA/$ds --method $method --seed $seed --robustness $COMMON
    done
    for m in $M; do
      done_already "${ds}__unimodal__${m}" $seed || \
        python train.py --data $DATA/$ds --method unimodal --unimodal $m --seed $seed $COMMON
    done
  done
done

# ---- Table V (ablation)
M=$(mods_of $ABL)
for seed in $SEEDS; do
  for m in $M; do
    keep=$(without "$M" $m)
    python train.py --data $DATA/$ABL --method mmrl --modalities $keep --seed $seed $COMMON
  done
  for ab in fusion attention mafe; do
    python train.py --data $DATA/$ABL --method mmrl --ablate $ab --seed $seed $COMMON
  done
  python train.py --data $DATA/$ABL --method mmrl --lambda_align 0 --seed $seed $COMMON
done

# ---- Table VI (cost), one dataset with all modalities present
COSTDS=${COSTDS:-arxiv_mml}
for method in mmrl cma mmbert late early; do
  python profile_cost.py --data $DATA/$COSTDS --method $method --out $OUT/cost
done
python profile_cost.py --data $DATA/$COSTDS --method unimodal --unimodal text --out $OUT/cost

# ---- statistics, tables, figures
python stats.py --results $OUT --out $OUT/stats.json
python make_tables.py --results $OUT --ablation_dataset $ABL --roc_dataset $ROC \
  --retrieval_dataset biomed_retrieval --curve_dataset $ABL --out paper_out
echo "Send paper_out/summary.json, paper_out/tables.md and $OUT/stats.json for the Section V rewrite."
