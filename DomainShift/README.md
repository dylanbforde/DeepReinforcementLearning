# Deep Reinforcement Learning with Domain Shift Prediction

This project investigates how deep reinforcement learning agents can adapt to domain shifts using a domain shift prediction mechanism. The implementation allows testing on different environments (BipedalWalker and CartPole) with and without domain shift prediction enabled.

## Project Structure

The code is organized into four main branches:

1. `full_code_bipedal_gcp`: BipedalWalker environment with domain shift predictor, adapted for GCP
2. `control_code_bipedal_gcp`: BipedalWalker environment without domain shift predictor (control), adapted for GCP
3. `full_code_cartpole_gcp`: CartPole environment with domain shift predictor, adapted for GCP
4. `control_code_cartpole_gcp`: CartPole environment without domain shift predictor (control), adapted for GCP

## Setup Instructions

### Local Environment

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DeepReinforcementLearning
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Google Cloud Platform

For running experiments on GCP with GPU acceleration, see [GCP_SETUP.md](GCP_SETUP.md) for detailed instructions.

## Usage

### Running BipedalWalker Experiments

**With Domain Shift Predictor:**
```bash
git checkout full_code_bipedal_gcp
cd DomainShift
python Main.py --trials 10 --episodes 1000
```

**Control (Without Domain Shift Predictor):**
```bash
git checkout control_code_bipedal_gcp
cd DomainShift
python Main.py --trials 10 --episodes 1000
```

### Running CartPole Experiments

**With Domain Shift Predictor:**
```bash
git checkout full_code_cartpole_gcp
cd DomainShift
python CartPoleMain.py --trials 10 --episodes 500
```

**Control (Without Domain Shift Predictor):**
```bash
git checkout control_code_cartpole_gcp
cd DomainShift
python CartPoleMain.py --trials 10 --episodes 500
```

### Command Line Arguments

- `--cloud`: Run in cloud mode without visualization (for GCP)
- `--trials`: Number of trials to run (default: 10)
- `--episodes`: Max number of episodes per trial (default: 1000 for BipedalWalker, 500 for CartPole)

## Analyzing Results

After running experiments, use the analysis script to compare results:

```bash
python analyze_results.py
```

This will generate comparison plots and performance statistics across all experiment types.

## Project Description

We aim to use a deep learning neural network approach to not just detect, but numerically quantify domain shifts experienced by a reinforcement learning (RL) agent. In order to achieve this, we implement a domain shift prediction module that estimates the suitability of the current policy for the current environment state.

The domain shift prediction mechanism works by:
1. Analyzing both the state and a quantified domain shift metric
2. Predicting a suitability score between 0 and 1
3. When suitability is low, increasing exploration to adapt to the new environment

This approach helps the agent adapt more quickly to changing environments, leading to faster convergence and better performance under domain shifts.

## References

1. Alexander Steinparz, C., Schmied, T., Paischer, F., Dinu, M., Prakash Patil, V., Bitto-Nemling, A., Eghbal-zadeh, H., Hochreiter, S. (2022). Reactive Exploration to Cope with Non-Stationarity in Lifelong Reinforcement Learning. Conference on Lifelong Learning Agents. https://arxiv.org/abs/2207.05742

2. Deepak Pathak, Pulkit Agrawal, Alexei A. Efros and Trevor Darrell. Curiosity-driven Exploration by Self-supervised Prediction. In ICML 2017. https://pathak22.github.io/noreward-rl/

3. Hyperparameter scheduling. https://paperswithcode.com/methods/category/learning-rate-schedules