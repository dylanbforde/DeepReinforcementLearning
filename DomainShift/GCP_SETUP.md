# Google Cloud Platform Setup Guide

This guide explains how to set up and run the domain shift experiments on Google Cloud Platform.

## Prerequisites

1. Google Cloud Platform account with billing enabled
2. Basic familiarity with GCP Compute Engine
3. gcloud CLI installed (optional)

## Setting up a Compute Engine VM

1. Go to the Google Cloud Console
2. Navigate to Compute Engine > VM instances
3. Click "Create Instance"
4. Configure your VM:
   - Name: `rl-domain-shift`
   - Region: Choose a region close to you
   - Machine type: Select a machine with at least 4 vCPUs and 16GB memory
   - Boot disk: Ubuntu 20.04 LTS (or newer)
   - Boot disk size: At least 50GB 
   - Check "Enable access to NVIDIA GPU"
   - GPU type: Select NVIDIA T4 or better
   - Firewall: Allow HTTP/HTTPS traffic
5. Click "Create"

## Setting up the Environment

1. SSH into your VM:
   ```
   gcloud compute ssh rl-domain-shift
   ```

2. Install CUDA and necessary packages:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-venv git
   
   # Install CUDA following NVIDIA instructions
   # Instructions may vary based on Ubuntu version
   # See: https://developer.nvidia.com/cuda-downloads
   ```

3. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/DeepReinforcementLearning.git
   cd DeepReinforcementLearning
   ```

4. Set up virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```

## Running the Experiments

There are four different experimental setups, each with its own branch:

### 1. BipedalWalker with Domain Shift Predictor

```bash
git checkout full_code_bipedal_gcp
cd DomainShift
./run_on_gcp.sh
```

### 2. BipedalWalker without Domain Shift Predictor (Control)

```bash
git checkout control_code_bipedal_gcp
cd DomainShift
./run_control_on_gcp.sh
```

### 3. CartPole with Domain Shift Predictor

```bash
git checkout full_code_cartpole_gcp
cd DomainShift
./run_cartpole_on_gcp.sh
```

### 4. CartPole without Domain Shift Predictor (Control)

```bash
git checkout control_code_cartpole_gcp
cd DomainShift
./run_cartpole_control_on_gcp.sh
```

## Downloading Results

After the experiments complete, you'll find compressed results in `.tar.gz` files. Download these files to your local machine:

```bash
# From your local machine, not the VM
gcloud compute scp rl-domain-shift:~/DeepReinforcementLearning/DomainShift/results*.tar.gz .
```

## Analyzing Results

1. Extract the downloaded archives:
   ```bash
   tar -xzf results.tar.gz
   tar -xzf results_control.tar.gz
   tar -xzf results_cartpole.tar.gz
   tar -xzf results_cartpole_control.tar.gz
   ```

2. Each results directory contains:
   - CSV logs of all trials
   - Trained model files (.pth)
   - Optuna study databases
   - Summary files

3. You can use the analysis scripts to compare performance:
   ```bash
   python analyze_results.py
   ```

## Additional Notes

- You can adjust the number of trials and episodes in each run script
- Set the `CLOUD_MODE` environment variable to `true` to disable visualization
- Use Optuna dashboards to visualize hyperparameter optimization: `optuna-dashboard storage.db`
- Remember to shut down your VM when not in use to avoid unnecessary charges