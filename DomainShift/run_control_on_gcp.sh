#!/usr/bin/env bash
set -euo pipefail

python DomainShift/Main.py --env bipedal --mode control --cloud --trials 10 --episodes 1000 --output-dir results/bipedal_control
tar -czf bipedal_control_results.tar.gz results/bipedal_control
