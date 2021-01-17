#!/usr/bin/env python

from setuptools import setup, find_packages


dependencies = [
            "botorch",
            "configargparse",
            "icecream",
            "imageio",
            "matplotlib",
            "numpy",
            "opencv_python",
            "scipy",
            "scikit-optimize",
            "sklearn",
            "tensorboardX",
            "torchdiffeq",
            "torchvision",
            "torchviz",
            "zarr",
            "botorch"
        ]

setup(name='cubeadv',
      version="0.1",
      packages=find_packages(exclude=["notebooks", "scripts"]),
      package_dir={"":"./"},
      install_requires=dependencies)
