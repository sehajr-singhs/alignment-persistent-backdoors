@echo off
cd /d C:\Users\sehaj\OneDrive\Desktop\repos\alignment-persistent-backdoors
echo Starting NMI experiments at %TIME% > quick_run.log
python -u run_quick_nmi.py >> quick_run.log 2>&1
echo Finished at %TIME% >> quick_run.log
