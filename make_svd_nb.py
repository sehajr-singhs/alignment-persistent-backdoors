"""Generate SVD analysis Kaggle notebook."""
import json, os, base64

with open("nmi_gpu_analysis.py") as f:
    script = f.read()

md = (
    "# NMI SVD Entanglement + Orthogonal Intervention (GPU, ~30 min)\n\n"
    "Runs on P100/T4 GPU. SVD cosine similarity, superposition scores, "
    "orthogonal projection intervention, dispersed attacker.\n"
)

# Force GPU
pre = (
    "import os\n"
    "os.environ.pop('HF_HUB_OFFLINE', None)\n"
    "import subprocess, sys\n"
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "    'transformers>=4.45', 'peft>=0.13', 'accelerate'])\n"
    "print('Dependencies installed')\n\n"
)

# Embed script via base64
bs = script.encode("ascii", errors="replace")
b64str = base64.b64encode(bs).decode()
chunk = 3000
lines = [pre, "import base64, sys\nb64 = (\n"]
for i in range(0, len(b64str), chunk):
    lines.append(f'    "{b64str[i:i+chunk]}"\n')
lines.append(
    ")\n"
    'script = base64.b64decode("".join(b64)).decode()\n'
    "print(f'Running SVD analysis ({len(script)} bytes)...')\n"
    "exec(script)\n"
)

pkg = (
    "import zipfile, json, os\n"
    "if os.path.exists('results/nmi'):\n"
    "    files = sorted(os.listdir('results/nmi'))\n"
    "    with zipfile.ZipFile('svd_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:\n"
    "        for f in files:\n"
    "            fp = os.path.join('results/nmi', f)\n"
    "            if os.path.isfile(fp):\n"
    "                z.write(fp)\n"
    "    print(f'Packaged {len(files)} files into svd_results.zip')\n"
)

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [md]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [pkg]},
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

outpath = "kaggle/nmi_svd.ipynb"
os.makedirs("kaggle", exist_ok=True)
with open(outpath, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {outpath} ({os.path.getsize(outpath)/1024:.0f} KB)")
