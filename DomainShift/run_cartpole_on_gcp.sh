#!/usr/bin/env bash
set -euo pipefail

python DomainShift/Main.py --env cartpole --mode dsp --cloud --trials 10 --episodes 500 --output-dir results/cartpole_dsp
tar -czf cartpole_dsp_results.tar.gz results/cartpole_dsp
