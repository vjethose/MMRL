#!/usr/bin/env bash
# One Kaggle (or Colab) GPU session. Run a STAGE, save the notebook version,
# then attach its output to the next session and run the next STAGE.
#   STAGE=prep      download + convert the three datasets, embed KG entities
#   STAGE=speed     2-minute timing check (do this first!)
#   STAGE=sqa       main comparison on ScienceQA
#   STAGE=arxiv     main comparison on ogbn-arxiv
#   STAGE=roco      main comparison on ROCOv2
#   STAGE=ablation  Table V on ScienceQA
#   STAGE=cost      Table VI timing on this GPU
#   STAGE=report    stats, tables, figures, Section V draft
# Every run skips results that already exist, so a session that hits the
# 12-hour limit can simply be repeated.
set -uo pipefail
cd "$(dirname "$0")/.."

STAGE=${STAGE:?set STAGE}
WORK=${WORK:-/kaggle/working}
DATA=${DATA:-$WORK/data}
OUT=${OUT:-$WORK/results}
PREV=${PREV:-}                       # folder of an earlier session's output to resume from
SEEDS=${SEEDS:-"0 1 2"}
AMP=${AMP:-fp16}                     # T4/P100: fp16. A100: bf16
COMMON="--preset ${PRESET:-paper} --amp $AMP --workers ${WORKERS:-2} --epochs ${EPOCHS:-10} --patience ${PATIENCE:-3} --bs ${BS:-16} --accum ${ACCUM:-2} --out $OUT"
mkdir -p "$DATA" "$OUT"
if [ -n "$PREV" ]; then
  [ -d "$PREV/results" ] && cp -rn "$PREV/results/." "$OUT/" 2>/dev/null
  [ -d "$PREV/data" ] && [ ! -f "$DATA/scienceqa/meta.json" ] && cp -rn "$PREV/data/." "$DATA/" 2>/dev/null
fi

name_of() {  # mirrors train.run_name for the skip check
  local ds=$1 method=$2 extra=$3 seed=$4
  echo "$OUT/${ds}__${method}${extra}__s${seed}.json"
}
START_TS=${START_TS:-$(date +%s)}
BUDGET_MIN=${BUDGET_MIN:-0}             # stop launching new runs after this many minutes (0 = no limit)
run() {  # ds method extra-name seed args...
  local ds=$1 method=$2 extra=$3 seed=$4; shift 4
  local f; f=$(name_of "$ds" "$method" "$extra" "$seed")
  if [ -f "$f" ]; then echo "skip $f"; return; fi
  if [ "$BUDGET_MIN" -gt 0 ] && [ $(( ($(date +%s) - START_TS) / 60 )) -ge "$BUDGET_MIN" ]; then
    echo "budget reached, not starting $f"; return; fi
  echo "=== $ds $method$extra seed $seed ($(date +%H:%M))"
  python train.py --data "$DATA/$ds" --method "$method" --seed "$seed" $COMMON "$@" || echo "FAILED $ds $method$extra $seed"
}
main_grid() {
  local ds=$1; shift
  local mods; mods=$(python -c "import json;print(' '.join(json.load(open('$DATA/$ds/meta.json'))['modalities']))")
  for seed in $SEEDS; do
    for method in mmrl cma mmbert late early; do run "$ds" "$method" "" "$seed" --robustness; done
    # a single-modality model cannot embed both text queries and image documents
    if [ "$(python -c "import json;print(json.load(open('$DATA/$ds/meta.json'))['task'])")" != retrieval ]; then
      for m in $mods; do run "$ds" unimodal "__$m" "$seed" --unimodal "$m"; done
    fi
  done
}

case $STAGE in
prep)
  mkdir -p "$WORK/raw"; pip install -q huggingface_hub
  # ScienceQA (official splits) from the Hugging Face hub
  python -c "from huggingface_hub import snapshot_download as s; s('derek-thomas/ScienceQA', repo_type='dataset', local_dir='$WORK/raw/sqa', allow_patterns=['data/*'])"
  python tools/prepare_scienceqa.py --src $WORK/raw/sqa/data --out "$DATA/scienceqa"
  # ogbn-arxiv: OGB files + raw titles/abstracts
  pip install -q ogb
  python -c "from ogb.nodeproppred import NodePropPredDataset as D; D('ogbn-arxiv', root='$WORK/raw/ogb')" <<< "y"
  wget -q -O $WORK/raw/titleabs.tsv.gz https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
  python tools/prepare_ogbn_arxiv.py --ogb_root $WORK/raw/ogb/ogbn_arxiv --titleabs $WORK/raw/titleabs.tsv.gz --out "$DATA/ogbn_arxiv"
  # ROCOv2: attach the Zenodo release (record 10821435) as a Kaggle dataset first, then point ROCO_ROOT at it
  if [ -n "${ROCO_ROOT:-}" ]; then
    python tools/prepare_rocov2.py --root "$ROCO_ROOT" --out "$DATA/rocov2" --copy_images
  else
    echo "ROCO_ROOT not set: skipping ROCOv2 (see kaggle/KAGGLE.md)"
  fi
  for ds in scienceqa rocov2; do [ -f "$DATA/$ds/kg/entities.json" ] && python tools/embed_entities.py "$DATA/$ds"; done
  for ds in scienceqa ogbn_arxiv rocov2; do [ -f "$DATA/$ds/meta.json" ] && python tools/check_dataset.py "$DATA/$ds"; done
  ;;
speed)
  python train.py --data "$DATA/scienceqa" --method mmrl --seed 99 --amp $AMP --workers 2 --epochs 1 --bs ${BS:-16} --accum 1 \
     --max_train 640 --out $WORK/speedtest
  ;;
sqa)   main_grid scienceqa ;;
arxiv) main_grid ogbn_arxiv ;;
roco)  main_grid rocov2 ;;
ablation)
  for seed in $SEEDS; do
    for ab in fusion attention mafe; do run scienceqa mmrl "__wo_$ab" "$seed" --ablate "$ab"; done
    run scienceqa mmrl "__wo_skga" "$seed" --lambda_align 0
    run scienceqa mmrl "__mods_text" "$seed" --modalities text
    run scienceqa mmrl "__mods_image" "$seed" --modalities image
  done
  ;;
cost)
  for ds in scienceqa ogbn_arxiv; do
    for method in mmrl cma mmbert late early; do
      python profile_cost.py --data "$DATA/$ds" --method $method --preset ${PRESET:-paper} --amp $AMP --out "$OUT/cost_$ds"
    done
    python profile_cost.py --data "$DATA/$ds" --method unimodal --unimodal text --preset ${PRESET:-paper} --amp $AMP --out "$OUT/cost_$ds"
  done
  ;;
report)
  python stats.py --results "$OUT" --out "$OUT/stats.json"
  python make_tables.py --results "$OUT" --cost "$OUT/cost_scienceqa" --ablation_dataset scienceqa --roc_dataset ogbn_arxiv \
     --retrieval_dataset rocov2 --curve_dataset scienceqa --confusion_dataset ogbn_arxiv --out $WORK/paper_out
  python tools/draft_section_v.py $WORK/paper_out/summary.json > $WORK/paper_out/section_v.md
  cp "$OUT/stats.json" $WORK/paper_out/
  cd "$WORK" && zip -qr paper_out.zip paper_out && echo "Download $WORK/paper_out.zip"
  ;;
*) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
