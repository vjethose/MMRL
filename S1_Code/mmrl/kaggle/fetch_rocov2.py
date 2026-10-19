#!/usr/bin/env python
"""Download ROCOv2 from Zenodo (latest version of record 10821435) and extract
only a seed-0 sample of images per split, so the full 4.6 GB training archive
never has to be unpacked.

  python kaggle/fetch_rocov2.py --out /tmp/rocov2 --n_train 10000 --n_eval 2000
"""
import argparse
import csv
import json
import os
import random
import urllib.request
import zipfile

API = "https://zenodo.org/api/records/10821435"


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def download(url, dst):
    if os.path.exists(dst):
        return
    print("downloading", os.path.basename(dst), flush=True)
    tmp = dst + ".part"
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
    os.rename(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rec = get_json(API)
    latest = rec.get("links", {}).get("latest")
    if latest:
        try:
            rec = get_json(latest)
        except Exception as e:  # fall back to the named record
            print("could not follow latest:", e)
    print("ROCOv2 record", rec.get("id"), rec.get("metadata", {}).get("version"), flush=True)
    files = {f["key"]: f["links"].get("self") or f["links"].get("content") for f in rec["files"]}
    json.dump({"record_id": rec.get("id"), "doi": rec.get("doi"), "files": sorted(files)}, open(os.path.join(a.out, "zenodo_record.json"), "w"), indent=1)
    rng = random.Random(a.seed)
    for split, n in (("train", a.n_train), ("valid", a.n_eval), ("test", a.n_eval)):
        for kind in ("captions", "concepts"):
            k = f"{split}_{kind}.csv"
            if k in files:
                download(files[k], os.path.join(a.out, k))
        with open(os.path.join(a.out, f"{split}_captions.csv")) as f:
            ids = [row[0] for i, row in enumerate(csv.reader(f)) if i > 0]
        zkey = f"{split}_images.zip"
        zpath = os.path.join(a.out, zkey)
        download(files[zkey], zpath)
        with zipfile.ZipFile(zpath) as z:
            members = {os.path.splitext(os.path.basename(m))[0]: m for m in z.namelist() if not m.endswith("/")}
            pool = sorted(i for i in ids if i in members)
            rng.shuffle(pool)
            keep = pool[:n]
            dst = os.path.join(a.out, f"{split}_images")
            os.makedirs(dst, exist_ok=True)
            for i in keep:
                with z.open(members[i]) as src, open(os.path.join(dst, os.path.basename(members[i])), "wb") as out:
                    out.write(src.read())
        os.remove(zpath)
        print(split, len(keep), "images extracted", flush=True)
    if "cui_mapping.csv" in files:
        download(files["cui_mapping.csv"], os.path.join(a.out, "cui_mapping.csv"))


if __name__ == "__main__":
    main()
