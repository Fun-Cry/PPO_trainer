# PPO Trainer

This script is designed to train an AI agent using Proximal Policy Optimization (PPO) with multi-processing. The default environment is the "ALE/Alien-v5" from the OpenAI Gym, but users can modify the script to train on any other environment. The AI agent's architecture should be defined in the `AlienBot` class in the `model.py` file.

## Prerequisites

Before running the script, make sure to prepare your environment accordingly:

1. Create the environment:

   ```shell
   conda env create -f environment.yml
   ```

2. Activate the `PPO`:

   ```shell
   conda activate PPO
   ```

3. Install PyTorch. If you already installed it in your base environment, run this command:

   ```shell
   conda install --channel defaults --override-channels pytorch
   ```

4. Install the required Gym environment. For example, if you want to use the "Alien-v5" environment, you can install it using the following command:

   ```shell
   pip install "gymnasium[atari, accept-rom-license]"
   ```

## Usage

After setting up the environment and dependencies, run the script using Python:

```shell
python trainer.py
```
