"""Create Kaggle notebook with P100-compatible install + subprocess execution."""
import json, os, base64

# Read the full experiment script
with open("nmi_gpu_full.py", "r") as f:
    script = f.read()

# Install cell - P100 compatible + torchao fix
install_cell = """# Cell 1: Install GPU-compatible packages
import subprocess, sys, os, importlib

import torch
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'
cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
print(f'GPU: {gpu_name} (SM {cap[0]}.{cap[1]})')

# Test CUDA works
cuda_ok = False
cuda_broken = False
if torch.cuda.is_available():
    try:
        _ = torch.zeros(1).cuda()
        cuda_ok = True
        print(f'CUDA OK: {torch.version.cuda}')
    except Exception as e:
        cuda_broken = True
        print(f'CUDA broken: {e}')

if cuda_broken or not torch.cuda.is_available():
    print('Need compatible PyTorch for this GPU...')
    for torch_ver, torch_url in [
        ('2.5.1', 'https://download.pytorch.org/whl/cu124'),
        ('2.4.1', 'https://download.pytorch.org/whl/cu121'),
        ('2.3.1', 'https://download.pytorch.org/whl/cu121'),
    ]:
        try:
            print(f'  Trying torch=={torch_ver}...')
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-q',
                f'torch=={torch_ver}', 'torchvision', 'torchaudio',
                '--index-url', torch_url, '--force-reinstall', '--no-deps'
            ], timeout=300)
            importlib.reload(torch)
            _ = torch.zeros(1, device='cuda')
            print(f'  SUCCESS with torch {torch_ver}! CUDA={torch.version.cuda}')
            cuda_ok = True
            break
        except Exception as e:
            print(f'  Failed: {e}')

# CRITICAL: Install peft==0.11.1 to avoid torchao dependency conflict
# peft>=0.12 requires torchao>=0.16, but Kaggle ships torchao==0.10
# peft 0.11.1 works fine and has no torchao requirement
print('Installing peft==0.11.1 (no torchao dependency)...')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'peft==0.11.1', 'transformers>=4.45,<5.0', 'accelerate>=0.26'],
    timeout=120)

try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'bitsandbytes'],
        timeout=60)
except Exception:
    print('bitsandbytes not available')

print('Dependencies installed!')
if torch.cuda.is_available():
    try:
        _ = torch.zeros(1).cuda()
        print(f'CUDA verified: {torch.cuda.get_device_name(0)}')
    except Exception as e:
        print(f'CUDA check failed: {e}')
else:
    print('WARNING: No CUDA - experiments will be slow')"""

# Embed script as base64 for portability
script_b64 = base64.b64encode(script.encode()).decode()

run_cell = f"""# Cell 2: Run full NMI experiment suite
import subprocess, sys, os, base64, time

# Decode and write the experiment script
script_b64 = '{script_b64}'
script = base64.b64decode(script_b64).decode()
with open('nmi_experiment.py', 'w') as f:
    f.write(script)

import torch
if torch.cuda.is_available():
    try:
        _ = torch.zeros(1).cuda()
        print('GPU Ready: ' + torch.cuda.get_device_name(0))
    except:
        print('WARNING: CUDA broken, will run on CPU')
else:
    print('WARNING: No GPU, running on CPU (will take hours)')

print('Starting NMI experiment suite...')
print('=' * 60)
t0 = time.time()

result = subprocess.run(
    [sys.executable, '-u', 'nmi_experiment.py'],
    timeout=7200,
)

elapsed = time.time() - t0
print(f'\\nExperiment completed in {{elapsed/60:.1f}} minutes')
print(f'Exit code: {{result.returncode}}')"""

package_cell = """# Cell 3: Package results for download
import zipfile, os, json

if os.path.exists('nmi_results'):
    files = sorted(os.listdir('nmi_results'))
    with zipfile.ZipFile('nmi_results.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            fp = os.path.join('nmi_results', f)
            if os.path.isfile(fp):
                zf.write(fp)
    print(f'Packaged {len(files)} result files into nmi_results.zip')
    print('\\n--- RESULTS SUMMARY ---')
    for f in files:
        if f.endswith('.json'):
            fp = os.path.join('nmi_results', f)
            try:
                d = json.load(open(fp))
                if 'baseline' in d:
                    b = d['baseline']
                    dpo = d.get('dpo', {}).get('after', {})
                    print(f'  {f}: ASR={b.get("asr",0):.3f} benign={b.get("benign_acc",0):.3f} DPO_ASR={dpo.get("asr",0):.3f}')
                elif 'total_time_seconds' in d:
                    print(f'  {f}: {d.get("n_experiments",0)} experiments in {d["total_time_seconds"]/60:.1f}min')
            except:
                print(f'  {f} (binary or error)')
    print('\\nDownload nmi_results.zip from the Output section below')
else:
    print('No results directory found. Check output above for errors.')"""

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# NMI Backdoor Circuit — Full GPU Experiment Suite\n",
            "\n",
            "Tests whether preference optimization (DPO) strengthens backdoors in instruction-tuned LLMs.\n",
            "\n",
            "**Experiments:**\n",
            "- 4 models (Qwen2.5-0.5B, SmolLM2-360M, Qwen2.5-1.5B, 7B QLoRA)\n",
            "- 5 seeds per model × 2 tasks (synthetic + code completion) = 40 runs\n",
            "- DPO persistence, surgical pruning, adaptive trigger, circuit analysis\n",
            "\n",
            "**Requirements:** Kaggle GPU (T4 or P100). ~50 min on T4, ~15 min on A100."
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
os.makedirs("kaggle", exist_ok=True)
with open(outpath, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {outpath} ({os.path.getsize(outpath)/1024:.0f} KB)")
