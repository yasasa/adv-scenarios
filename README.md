# Generating Transferable Adversarial Simulation Scenarios For Self-Driving

## Prerequisites

- [PyTorch](https://pytorch.org/get-started/locally/)
- [Kaolin Wisp](https://github.com/yasasa/kaolin-wisp.git)

For testing the simulation on CARLA, we will need a standalone instance of CARLA 0.9.11, as well as the
associated python packages.

Clone this repository(recursively), go inside the directory and possibly inside a virtualenv or conda, run:

```
pip install -r requirements.txt
pip install -e .
```

Download the models from [here]()

## How to use

To train a policy use: `scripts/train_simulator_policy.py`

To run an optimization on an existing policy use: `scripts/experiments.py`

To run the experiment with 2 cars and 3 hydrants turning right:

``` python scripts/experiment.py --cfg configs/custom_train_configs/gradient-attacks/NGP-ColourAttack-0-1-2car-3hydrant.txt ```

For the experiment with 2 cars and 3 hydrants going straight:

``` python scripts/experiment.py --cfg configs/custom_train_configs/gradient-attacks/NGP-ColourAttack-1-2-2car-3hydrant.txt ```

For the experiment with 2 cars and 3 hydrants going left:

``` python scripts/experiment.py --cfg configs/custom_train_configs/gradient-attacks/NGP-ColourAttack-1-0-2car-3hydrant.txt ```
