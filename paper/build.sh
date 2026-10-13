#!/bin/sh
# Regenerate every figure and table from ../results, audit the numbers in the prose, and compile the paper.
set -e
cd "$(dirname "$0")/.."
python paper/scripts/make_figures.py
python paper/scripts/make_tables.py
python paper/scripts/check_numbers.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
