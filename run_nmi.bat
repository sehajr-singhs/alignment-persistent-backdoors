@echo off
cd /d "C:\Users\sehaj\OneDrive\Desktop\repos\alignment-persistent-backdoors"
python -u run_nmi_critical.py > nmi_critical.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> nmi_critical.log
