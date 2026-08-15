#!/usr/bin/env bash
# Produce every deliverable from the committed results:
#   figures -> paper numbers -> both PDFs -> website
set -e
cd "$(dirname "$0")"
export PYTHONPATH=src

echo "=== figures ==="
python make_figures.py

echo "=== paper numbers ==="
python make_paper_numbers.py

echo "=== compile papers (two passes for cross-refs) ==="
cd paper
pdflatex -interaction=nonstopmode manuscript.tex > /dev/null
pdflatex -interaction=nonstopmode manuscript.tex > /dev/null
pdflatex -interaction=nonstopmode ieee_manuscript.tex > /dev/null
pdflatex -interaction=nonstopmode ieee_manuscript.tex > /dev/null
cd ..

echo "=== website ==="
python make_site.py

echo "=== done ==="
ls -la paper/*.pdf figs/*.pdf index.html
