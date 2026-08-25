"""Create Kaggle notebook from the experiment script."""
import json

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": ["# NMI Backdoor Circuit Full GPU Suite\n", "Runs on T4 GPU (~50 min)."]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import subprocess, sys\n", "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers>=4.45', 'peft', 'accelerate'])\n", "try:\n", "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'bitsandbytes'])\n", "except Exception:\n", "    print('bitsandbytes not available')\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import urllib.request\n", "url = 'https://raw.githubusercontent.com/sehajr-singhs/alignment-persistent-backdoors/main/nmi_gpu_full.py'\n", "urllib.request.urlretrieve(url, 'nmi_gpu_full.py')\n", "print('Downloaded')\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["exec(open('nmi_gpu_full.py').read())\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import zipfile, os\n", "with zipfile.ZipFile('nmi_results.zip', 'w') as zf:\n", "    for f in os.listdir('nmi_results'):\n", "        zf.write(os.path.join('nmi_results', f))\n", "print('Done')\n"]}
]

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}},
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("kaggle/nmi_full_gpu.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Wrote ipynb")
