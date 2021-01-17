import torch
from .raw_sensor import Sensor


class SensorRig(Sensor):
    def __init__(self):
        self.sensors = []
        self.rotations = []
        
    def add_sensor(self, sensor : Sensor, offset : torch.Tensor):
        self.sensors.append((offset, sensor))
    
    def read(self, field, state):
        ret = []
        for offset, sensor in self.sensors:
            new_state = state + offset
            ret.append(sensor.read(field, new_state))
        return ret
        