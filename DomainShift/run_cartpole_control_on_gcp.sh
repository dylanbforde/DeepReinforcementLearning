#!/usr/bin/env bash
set -euo pipefail

python DomainShift/Main.py --env cartpole --mode control --cloud --trials 10 --episodes 500 --output-dir results/cartpole_control
tar -czf cartpole_control_results.tar.gz results/cartpole_control
