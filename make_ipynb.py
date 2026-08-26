"""Create Kaggle notebook with P100-compatible install + subprocess execution."""
import json, os

# Read the full experiment script
with open("nmi_gpu_full.py", "r") as f:
    script = f.read()

# Install cell
install_cell = (
    "# Cell 1: Install P100-compatible packages\n"
    "import subprocess, sys\n"
    "\n"
    "import torch\n"
    "gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'\n"
    "cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)\n"
    "print(f'GPU: {gpu_name} (SM {cap[0]}.{cap[1]})')\n"
    "\n"
    "if torch.cuda.is_available() and cap[0] < 7:\n"
    "    print('Installing P100-compatible packages...')\n"
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "        'torch==2.3.1', 'torchvision==0.18.1', 'torchaudio==2.3.1',\n"
    "        '--index-url', 'https://download.pytorch.org/whl/cu121',\n"
    "        '--force-reinstall'])\n"
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "        'peft==0.11.1', 'transformers==4.45.2', 'accelerate>=0.26'])\n"
    "else:\n"
    "    print('Using system packages')\n"
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "        'transformers>=4.45', 'peft>=0.12', 'accelerate'])\n"
    "\n"
    "try:\n"
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'bitsandbytes'])\n"
    "except Exception:\n"
    "    print('bitsandbytes not available')\n"
    "print('Dependencies installed')"
)

# Run cell: write script to file, run via subprocess
# Use base64 to avoid quote escaping issues
import base64
script_b64 = base64.b64encode(script.encode()).decode()

run_cell = (
    "# Cell 2: Run experiment via subprocess\n"
    "import subprocess, sys, os, base64\n"
    "\n"
    "# Decode embedded script\n"
    "script_b64 = '" + script_b64 + "'\n"
    "script = base64.b64decode(script_b64).decode()\n"
    "with open('nmi_experiment.py', 'w') as f:\n"
    "    f.write(script)\n"
    "\n"
    "print('Starting NMI experiment suite (~50 min on GPU)...')\n"
    "print('=' * 60)\n"
    "\n"
    "result = subprocess.run(\n"
    "    [sys.executable, '-u', 'nmi_experiment.py'],\n"
    "    timeout=5400,\n"
    ")\n"
    "print(f'\\nExperiment exit code: {chr(123)}result.returncode{chr(125)}')"
)

# Package results cell
package_cell = (
    "# Cell 3: Package results\n"
    "import zipfile, os, json\n"
    "if os.path.exists('nmi_results'):\n"
    "    files = os.listdir('nmi_results')\n"
    "    with zipfile.ZipFile('nmi_results.zip', 'w') as zf:\n"
    "        for f in files:\n"
    "            zf.write(os.path.join('nmi_results', f))\n"
    "    print(f'Packaged {len(files)} result files')\n"
    "    print('Download nmi_results.zip from Output section below')\n"
    "    for f in sorted(files):\n"
    "        if f.endswith('.json'):\n"
    "            d = json.load(open(os.path.join('nmi_results', f)))\n"
    "            if 'asr' in d:\n"
    "                print(f'  {f}: ASR={d[\"asr\"]:.3f} benign={d.get(\"benign_acc\",0):.3f}')\n"
    "else:\n"
    "    print('No results found — check output above')"
)

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# NMI Backdoor Circuit — Full GPU Experiment Suite\n",
            "\n",
            "Runs on Kaggle T4/P100 GPU (~50 min). Tests:\n",
            "- 4 models x 5 seeds x 2 tasks\n",
            "- DPO persistence, surgical pruning, adaptive attacker, circuit analysis\n",
            "- Code completion (real task) + synthetic lookup\n",
            "- P100 compatible: auto-installs torch 2.3.1 if SM < 7.0"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [install_cell]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [run_cell]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [package_cell]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

outpath = "kaggle/nmi_full_gpu.ipynb"
with open(outpath, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {outpath} ({os.path.getsize(outpath)/1024:.0f} KB)")
