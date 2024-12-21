Branches of this repository are available as the following:
* Main: the most up to date branch being used.
* feature/DSP: this is the branch where tests are done when running domain shift predictor.
* feature/No-DSP: this is the branch where test are done when not running the predictor.


We aim to use a deep learning neural network approach to not just detect, but numerically quantify domain shifts experienced by an reinforcement learning (RL) agent. In order to achieve this the existing algorithm Reactive Exploration (which is itself based on an Intrinsic Curiosity Module algorithm) will be examined under new experimental conditions. Using metrics tracked by the algorithm, predictability of the distance between a current policy and its suitability to its new domain-shifted environment will be investigated.

The goal of this project is to establish if Reactive Exploration does have predictive capabilities for its current policy's performance in its new, domain-shifted environment. If it is capable of determining the distance between its policy and its new environment, this would open up avenues for agent-determined hyperparameter scheduling, or as an input to future meta-learning approaches. (Hyperparameter scheduling is a known method for quicker convergence of neural networks that takes into account the distance from optimality of the neural network as it converges).

Alexander Steinparz, C., Schmied, T., Paischer, F.,, Dinu, M., Prakash Patil, V., Bitto-Nemling, A., Eghbal-zadeh, H., Hochreiter, S. (2022). Reactive Exploration to Cope with Non-Stationarity in Lifelong Reinforcement Learning. Conference on Lifelong Learning Agents. https://arxiv.org/abs/2207.05742

Deepak Pathak, Pulkit Agrawal, Alexei A. Efros and Trevor Darrell. Curiosity-driven Exploration by Self-supervised Prediction. In ICML 2017. https://pathak22.github.io/noreward-rl/

Hyperparameter scheduling. https://paperswithcode.com/methods/category/learning-rate-schedules
