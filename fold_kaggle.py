"""Fold a completed Kaggle matrix run into the repo.

After `kaggle kernels output sehajrsingh/backdoors-survive-alignment-matrix`,
the downloaded `results.zip` (or an extracted folder) contains the matrix
JSONs. This script merges them into ./results/ (new files only, never
clobbering the pilot), then regenerates numbers, figures, and the site so
the papers pick up the scale-up numbers.

Usage:
    python fold_kaggle.py path/to/results.zip
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

RESULTS = Path("results")


def main(src: str) -> None:
    src = Path(src)
    tmp = Path("build_kaggle_fold")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
    else:
        shutil.copytree(src, tmp, dirs_exist_ok=True)

    new = 0
    for p in sorted(tmp.rglob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception as e:  # torn/corrupt artifact
            print(f"skip {p.name}: unparseable ({e})")
            continue
        dest = RESULTS / p.name
        if dest.exists():
            print(f"keep existing {p.name}")
            continue
        dest.write_text(json.dumps(data, indent=2))
        new += 1
        print(f"added {p.name}")

    shutil.rmtree(tmp)
    print(f"\n{new} new result file(s). Regenerating deliverables...")
    for step in ("make_paper_numbers.py", "make_figures.py", "make_site.py"):
        import subprocess
        r = subprocess.run([sys.executable, step], capture_output=True, text=True)
        print(f"  {step}: rc={r.returncode} {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
