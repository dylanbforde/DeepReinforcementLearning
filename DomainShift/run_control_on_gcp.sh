#!/bin/bash
# Script to run the control training (without domain shift predictor) on Google Cloud Platform

# Set environment variables
export CLOUD_MODE=true

# Create results directory
mkdir -p results

# Run the training with appropriate arguments
python Main.py --cloud --trials 10 --episodes 1000

# Compress results for easy download
tar -czf results_control.tar.gz results/

echo "Control training complete! Results are available in results_control.tar.gz"