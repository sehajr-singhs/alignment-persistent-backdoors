#!/usr/bin/env bash
# Usage: bash run_phase.sh LOG_NAME -- <phase args...>
# Runs one pipeline phase with the thread pool pinned (the reliable config on
# this machine) and tees output to logs/LOG_NAME.
set -u
cd /c/Users/sehaj/OneDrive/Desktop/repos/alignment-persistent-backdoors
mkdir -p logs
LOG="$1"; shift; shift   # drop LOG_NAME and the "--" separator
export PYTHONPATH=src
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 BACKDOOR_THREADS=12
echo "=== $LOG start $(date) ===" >> logs/"$LOG"
python -u "$@" >> logs/"$LOG" 2>&1
rc=$?
echo "=== $LOG end rc=$rc $(date) ===" >> logs/"$LOG"
exit $rc
