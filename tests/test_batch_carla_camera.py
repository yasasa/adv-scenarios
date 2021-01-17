import os
from pathlib import Path
import time
import signal

import pytest
import subprocess

import carla
import imageio
import torch

from cubeadv.sim.sensors import carla_camera
from test_utils import *

def try_connect_client(port):
    try:
        client = carla.Client("localhost", port)
        world = client.load_world('Town01')
        world = client.reload_world()
        return client, world
    except Exception as e:
        return None, None

@pytest.fixture
def carla_worlds():
    base_path = os.environ.get("CARLA_PATH")
    sim_path = Path(base_path, "CarlaUE4.sh")
    
    process_count = 6
    
    assert(sim_path.exists())
    proc = [subprocess.Popen(" /Game/Carla/Maps/Town01", executable=sim_path, preexec_fn=os.setsid) for _ in range(process_count)]
    
    try:
        worlds = [None]*process_count
        for attempt in range(10):
            time.sleep(2)
            for id in range(process_count):
                if worlds[id] is None:
                    client, world = try_connect_client(2000 + id)
                    worlds[id] = world
                    
        if not all(worlds):
            assert(False and "Failed to connect to carla")

        yield worlds
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

@pytest.mark.parametrize("location", [[105, 135, 0.]])
@image_comparison_test
def test_carla_camera(carla_worlds, location):
    location = torch.tensor(location)
        
    camera = carla_camera.BatchCarlaCamera(carla_worlds)
    location = location.expand(len(carla_worlds), 1)
    image = camera.read(location)
    
    return image
    
    
    