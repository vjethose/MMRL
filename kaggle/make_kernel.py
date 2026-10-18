#!/usr/bin/env python
"""Build a self-contained Kaggle script kernel for one stage.

The kernel carries the code as a base64 tarball, restores data.tar / results.tar
from the kernels it lists as sources, runs kaggle/session.sh for the stage, and
writes data.tar (prep), results.tar and logs to /kaggle/working.

  python kaggle/make_kernel.py --stage prep --user <kaggle-user> --out push_prep.json
"""
import argparse
import base64
import io
import json
import os
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TEMPLATE = r'''
import base64, glob, io, os, subprocess, tarfile, time, json
STAGES = __STAGES__
BUDGET_MIN = __BUDGET__
CODE = "__CODE__"
T0 = time.time()
W, T = "/kaggle/working", "/tmp/mmrl"
os.makedirs(T, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz").extractall(T)
os.chdir(T + "/mmrl")

def sh(cmd):
    print(">>", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode

sh("nvidia-smi || true")
sh("python -c \"import torch, transformers, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\" | tee " + W + "/environment.txt")
for t in sorted(glob.glob("/kaggle/input/*/data.tar")) + sorted(glob.glob("/kaggle/input/*/*/data.tar")):
    sh(f"tar -xf {t} -C {T}")
os.makedirs(T + "/results", exist_ok=True)
for t in sorted(glob.glob("/kaggle/input/*/results.tar")) + sorted(glob.glob("/kaggle/input/*/*/results.tar")):
    sh(f"tar -xf {t} -C {T} --skip-old-files")
env = dict(os.environ, WORK=T, DATA=T + "/data", OUT=T + "/results", AMP="fp16", WORKERS="2",
           START_TS=str(int(T0)), BUDGET_MIN=str(BUDGET_MIN), ROCO_ROOT=T + "/rocov2_raw")
for stage in STAGES:
    if stage == "prep":
        sh("pip install -q ogb huggingface_hub")
        sh(f"python kaggle/fetch_rocov2.py --out {T}/rocov2_raw > {W}/fetch_rocov2.log 2>&1; tail -5 {W}/fetch_rocov2.log")
    log = f"{W}/stage_{stage}.log"
    rc = subprocess.run(f"STAGE={stage} bash kaggle/session.sh 2>&1 | tee {log}", shell=True, env=env).returncode
    print("stage", stage, "exit", rc, flush=True)
if "prep" in STAGES:
    sh(f"cd {T} && tar -cf {W}/data.tar data")
    sh(f"cp {T}/rocov2_raw/zenodo_record.json {W}/ 2>/dev/null || true")
sh(f"cd {T} && tar -cf {W}/results.tar results")
if os.path.exists(T + "/paper_out.zip"):
    sh(f"cp {T}/paper_out.zip {W}/")
sh(f"ls {T}/results | head -200 > {W}/results_listing.txt; ls {T}/results | wc -l")
json.dump({"stages": STAGES, "minutes": round((time.time() - T0) / 60, 1)}, open(W + "/stage_summary.json", "w"))
'''


def code_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        def filt(ti):
            parts = ti.name.split("/")
            if any(p in ("data", "runs", "__pycache__") for p in parts[1:2]) or "__pycache__" in parts:
                return None
            return ti
        tf.add(ROOT, arcname="mmrl", filter=filt)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="comma list, e.g. prep,speed")
    ap.add_argument("--user", required=True)
    ap.add_argument("--slug", default=None)
    ap.add_argument("--sources", default="", help="comma list of kernel slugs (user/slug) whose output to mount")
    ap.add_argument("--budget_min", type=int, default=660)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stages = a.stage.split(",")
    slug = a.slug or "mmrl-" + "-".join(stages)
    src = TEMPLATE.replace("__STAGES__", json.dumps(stages)).replace("__BUDGET__", str(a.budget_min)).replace("__CODE__", code_b64())
    payload = {"slug": f"{a.user}/{slug}", "newTitle": slug, "text": src, "language": "python", "kernelType": "script",
               "isPrivate": True, "enableGpu": True, "enableTpu": False, "enableInternet": True,
               "datasetDataSources": [], "competitionDataSources": [],
               "kernelDataSources": [s for s in a.sources.split(",") if s], "categoryIds": []}
    json.dump(payload, open(a.out, "w"))
    print(a.out, f"{len(src) / 1024:.0f} KB source")


if __name__ == "__main__":
    main()
