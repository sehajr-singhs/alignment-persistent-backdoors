"""Create Kaggle notebook with FULL script embedded (no GitHub download needed)."""
import json, os

# Read the full experiment script
with open("nmi_gpu_full.py", "r") as f:
    script = f.read()

# Split into cells for readability
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# NMI Backdoor Circuit — Full GPU Experiment Suite\n",
            "\n",
            "Runs on Kaggle T4/P100 GPU (~50 min). Tests:\n",
            "- 4 models × 5 seeds × 2 tasks = 40 experiment runs\n",
            "- DPO persistence, surgical pruning, adaptive attacker, circuit analysis\n",
            "- Code completion (real task) + synthetic lookup\n",
            "- Cross-architecture: Qwen2.5-0.5B, SmolLM2-360M, Qwen2.5-1.5B, 7B QLoRA"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 1: Install dependencies\n",
            "import subprocess, sys\n",
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n",
            "    'transformers>=4.45', 'peft>=0.7', 'accelerate'])\n",
            "try:\n",
            "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'bitsandbytes'])\n",
            "except Exception:\n",
            "    print('bitsandbytes not available — 7B will be skipped')\n",
            "print('Dependencies installed')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 2: Verify GPU\n",
            "import torch\n",
            "print(f'CUDA: {torch.cuda.is_available()}')\n",
            "if torch.cuda.is_available():\n",
            "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
            "    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 3: Run full experiment suite\n",
            "# This cell contains the complete experiment script\n",
            script
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 4: Package results for download\n",
            "import zipfile, os, json\n",
            "if os.path.exists('nmi_results'):\n",
            "    with zipfile.ZipFile('nmi_results.zip', 'w') as zf:\n",
            "        for f in os.listdir('nmi_results'):\n",
            "            zf.write(os.path.join('nmi_results', f))\n",
            "    print(f'Packaged {len(os.listdir(\"nmi_results\"))} result files')\n",
            "    print('Download nmi_results.zip from the Output section →')\n",
            "else:\n",
            "    print('No results directory found')"
        ]
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
