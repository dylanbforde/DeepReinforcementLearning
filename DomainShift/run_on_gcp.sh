#!/usr/bin/env bash
set -euo pipefail

python DomainShift/Main.py --env bipedal --mode dsp --cloud --trials 10 --episodes 1000 --output-dir results/bipedal_dsp
tar -czf bipedal_dsp_results.tar.gz results/bipedal_dsp
