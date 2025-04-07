#!/bin/bash
# Script to run the CartPole training with domain shift predictor on Google Cloud Platform

# Set environment variables
export CLOUD_MODE=true

# Create results directory
mkdir -p results_cartpole

# Run the training with appropriate arguments
python CartPoleMain.py --cloud --trials 10 --episodes 500

# Compress results for easy download
tar -czf results_cartpole.tar.gz results_cartpole/

echo "CartPole training complete! Results are available in results_cartpole.tar.gz"