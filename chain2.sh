#!/usr/bin/env bash
# Remaining pilot phases, crash-safe: every phase skips if its result exists
# and every training phase resumes from its last checkpoint.
set -u
cd /c/Users/sehaj/OneDrive/Desktop/repos/alignment-persistent-backdoors
export PYTHONPATH=src
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 BACKDOOR_THREADS=12
mkdir -p logs

echo "=== chain2 start $(date) ==="

# Only one chain may run at a time: kill stale pipeline processes from any
# previous (possibly duplicated) launch before starting.
powershell -ExecutionPolicy Bypass -File kill_stale.ps1 2>/dev/null
sleep 5

# wait for the baseline (p=0.0) training result (written atomically-ish by
# the trainer; give it a beat once the file appears)
for i in $(seq 1 60); do
  if [ -f results/poison_0.0_1.json ]; then
    sleep 15
    break
  fi
  sleep 60
done

echo "=== [2] eval baseline ==="
python -u -m backdoors.run_all --evalsize 150 --phase eval --rates 0.0 >> logs/baseline_eval.log 2>&1 || echo "PHASE eval-baseline FAILED"

echo "=== [3] persist (clean FT on poisoned model) ==="
python -u -m backdoors.run_all --evalsize 150 --phase persist --rates 0.05 --steps 120 >> logs/persist.log 2>&1 || echo "PHASE persist FAILED"

echo "=== [4] unlearn (gradient ascent) ==="
python -u -m backdoors.run_all --evalsize 150 --phase unlearn --variant ascent --steps 120 >> logs/unlearn.log 2>&1 || echo "PHASE unlearn FAILED"

echo "=== [5] detect poisoned ==="
python -u -m backdoors.run_all --evalsize 150 --phase detect --rates 0.05 >> logs/detect.log 2>&1 || echo "PHASE detect FAILED"

echo "=== [6] detect baseline (contrast) ==="
python -u -c "from backdoors import detect; detect.run_detection(0.0, 1)" >> logs/detect_base.log 2>&1 || echo "PHASE detect-base FAILED"

echo "=== chain2 done $(date) ==="
