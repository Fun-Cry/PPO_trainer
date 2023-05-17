# PPO Trainer

This script is designed to train an AI agent using Proximal Policy Optimization (PPO) with multi-processing, the default environment is the "ALE/Alien-v5" from the OpenAI Gym, but user can modify to train on any other environment.
The AI agent's architecture should be defined in the `AlienBot` class in the `model.py` file.

## Usage

1. Make sure you have the necessary dependencies installed (e.g., `pytorch`, `gym`, `numpy`).

2. Define your AI agent in the `AlienBot` class in the `model.py` file.

3. Run the script using Python:

```shell
python trainer.py
```
