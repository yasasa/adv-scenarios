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

Download the models from [here](https://drive.google.com/file/d/1gbz2L_C78M9LRYG1fXafmnuG5-NinLWV/view?usp=sharing)

## How to use

To train a policy use: `scripts/train_simulator_policy.py`

To run an optimization on an existing policy use: `scripts/experiments.py`

To run the experiment with 2 cars and 3 hydrants turning right:

``` python scripts/experiment.py --cfg configs/custom_train_configs/gradient-attacks/NGP-ColourAttack-0-1-2car-3hydrant.txt ```

For the experiment with 2 cars and 3 hydrants going straight:

``` python scripts/experiment.py --cfg configs/custom_train_configs/gradient-attacks/NGP-ColourAttack-1-2-2car-3hydrant.txt ```

For the experiment with 2 cars and 3 hydrants going left:

``` python scripts/experiment.py --cfg configs/custom_train_configs/gradient-attacks/NGP-ColourAttack-1-0-2car-3hydrant.txt ```

Refer to the specific config files and the documentation inside the config parser for info about the cconfiguration parameters.

## Training models:
NeRF:
- To train the nerf model use [Kaolin Wisp](https://github.com/yasasa/kaolin-wisp.git). The specific kaolin wisp configs used for the CARLA setup is: `wisp/configs/ngp_nerf_bg_new.yaml`, for real world data we used the config `wisp/configs/ngp_my_580.yaml`.
- To generate CARLA data for training, the script `scripts/datacollector.py` can be used to uniformly sample a area in the CARLA map with images.
Policy:
- For training the neural network policies in CARLA and real world see the scripts `scripts/train_*_policy.py`. The CARLA verison of the script depends on a custom data format, and will collect data in addition to training the policy in loop. The real world version of the script requires the data and the controls, where the images are with the appropriate indexes, and a associated json file with the steering angle for that specific image.

## Visualization

For examples on how to visualize policy trajectories under different configurations(CARLA/NeRF), referto the test directory. We also using tensorboard logging inside the `scripts/experiment` file to output the trajectory of the vehicle on each iteration.


## Code Base Walk Through
### scripts/
This folder contains many utility scripts to run experiments and or train models, see above for the useful scripts.
### cubeadv/
This is the main package with our adversarial training code
#### cubeadv/sim/sensors
This folder contains all the utilities associated with wrapping sensors from different target simulators to the adversarial code. The target simulation modes are either instant-ngp or CARLA. In here we also contain majority of the code for blending multiple simulation formats to insert adversarial objects into the sensor feed.
#### cubeadv/sim/dynamics
Small set of utilites for different dynamics models to use
#### cubeadv/sim/simple_sim
Set of debug simulators only used for testing code agnostic of the true simulator
#### cubeadv/sim/utils
Contains code for holding map data(`path_map.py`) 
#### cubeadv/opt
Contains different optimizers, as well as a pytorch module to perform adjoint optimization on the forward shooting simulator, this saves memory by avoiding independent memory allocations for every node in the graph because our parameter size is much smaller than the extent of the graph.
#### cubeadv/fields
Thin wrapper around instant ngp to be used in thsi code base.
#### cubeadv/policies
Various different policies used in this code, with the main one being `*cnn.py`

To cite:

```
@misc{abeysirigoonawardena2024generating,
      title={Generating Transferable Adversarial Simulation Scenarios for Self-Driving via Neural Rendering}, 
      author={Yasasa Abeysirigoonawardena and Kevin Xie and Chuhan Chen and Salar Hosseini and Ruiting Chen and Ruiqi Wang and Florian Shkurti},
      year={2024},
      eprint={2309.15770},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
