"""Generate lean Kaggle notebook - uses exec(open().read()) to run nmi_lean.py inline."""
import json, os

with open("nmi_lean.py") as f:
    script = f.read()

md = (
    "# Lean NMI Experiment (CPU, ~15 min)\n\n"
    "Runs 0.5B model, 5 seeds x 2 tasks with DPO, pruning, circuit analysis.\n"
    "No GPU needed. Manual LoRA (no peft dependency)."
)

# Build the run cell with separate source lines
run_lines = [
    "# Force CPU before any torch import\n",
    "import os\n",
    "os.environ['CUDA_VISIBLE_DEVICES'] = ''\n",
    "os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'\n",
    "\n",
    "import torch\n",
    "torch.cuda.is_available = lambda: False\n",
    "torch.cuda.device_count = lambda: 0\n",
    "\n",
    "# Write the experiment script to a file and exec it\n",
    "import base64, sys\n",
    "b64 = (\n",
]

# Split the script into chunks of ~2000 chars for safe JSON
bs = script.encode()
b64str = __import__('base64').b64encode(bs).decode()
chunk_size = 2000
for i in range(0, len(b64str), chunk_size):
    run_lines.append(f'    "{b64str[i:i+chunk_size]}"\n')

run_lines.append(
    ")\n"
    "script = __import__(\"base64\").b64decode(\"\".join(b64)).decode()\n"
    "print(f'Running lean NMI experiment ({len(script)} bytes)...')\n"
    "exec(script)\n"
)

pkg_lines = [
    "import zipfile, os, json\n",
    "if os.path.exists('nmi_results'):\n",
    "    files = sorted(os.listdir('nmi_results'))\n",
    "    with zipfile.ZipFile('nmi_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:\n",
    "        for f in files:\n",
    "            fp = os.path.join('nmi_results', f)\n",
    "            if os.path.isfile(fp):\n",
    "                z.write(fp)\n",
    "    print(f'Packaged {len(files)} files')\n",
    "    for f in files:\n",
    '        if f.endswith(".json"):\n',
    '            d = json.load(open(os.path.join("nmi_results", f)))\n',
    '            b = d.get("baseline", {})\n',
    '            dp = d.get("dpo", {}).get("after", {})\n',
    '            print(f"  {f}: ASR={b.get(\'asr\',0):.3f} benign={b.get(\'benign_acc\',0):.3f} DPO={dp.get(\'asr\',0):.3f}")\n',
    "    print('\\nDownload nmi_results.zip from Output')\n",
]

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [md]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": run_lines},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": pkg_lines},
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

outpath = "kaggle/nmi_lean.ipynb"
os.makedirs("kaggle", exist_ok=True)
with open(outpath, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {outpath} ({os.path.getsize(outpath)/1024:.0f} KB)")
