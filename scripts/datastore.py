import numpy as np
import diskcache as dc

from typing import Any, List, Union, Tuple
from torch.utils.data import Dataset
import torch

import threading
import concurrent.futures
from functools import partial
import itertools

import pickle


class Datastore:
    def __init__(self, cache_size, path="dataset"):
        self.cache_size = cache_size # Not used currently
        self.cache = dc.Cache(path, eviction_policy='none', size_limit=int(1e10))
        if not 'size' in self.cache:
            self.cache['size'] = 0
        #self.cache['size'] = 0
        
        self.write_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.indices_on_flight = 0
        self.writes = []
    
    def __getitem__(self, key: int):
        label = self.cache[f"{key}_label"]
        data = self.cache[f"{key}_data"]
        type = self.cache[f"{key}_type"]
        shape = self.cache[f"{key}_shape"]
        
        data = [np.frombuffer(data_, dtype=np.dtype(type_)).reshape(*shape_) for data_, shape_, type_ in zip(data, shape, type)]
        if len(data) > 4:
            x, goal, image, depth, start = data
            data = (x, goal, image, start)
        label = pickle.loads(label)
        return label, *data
    
        
    def write_batch(self, labels, values):
        # Trick is to track the indices and defer the writes we will update the sizes
        # of the cache on sync
        size = self.size
        indices = list(range(size + self.indices_on_flight, 
                                size + self.indices_on_flight + len(labels)))
        self.indices_on_flight += len(labels)
        
        curr_writes = self.write_pool.map(self._set_item, indices, labels, *values)
        self.writes = itertools.chain(curr_writes, self.writes)
        
    def sync(self):
        writes = []
        for idx in self.writes:
            if idx is not None:
                writes.append(idx)
           
        self.cache['size'] += self.indices_on_flight
        assert(len(writes) == self.indices_on_flight)
        
        self.indices_on_flight = 0
        #concurrent.futures.wait(self.writes)       
        
    def _set_item(self, key, label, *values):
        self.cache[f"{key}_label"] = pickle.dumps(label)
        self.cache[f"{key}_data"] = [value_.tobytes() for value_ in values]
        self.cache[f"{key}_type"] = [value_.dtype.str for value_ in values] 
        self.cache[f"{key}_shape"] = [value_.shape for value_ in values]
        return key
      
    def __setitem__(self, key : int, value: Tuple[Any, List[np.array]]):
        self._set_item(key, value[0], value[1])
            
    @property
    def size(self):
        return self.cache['size']
            
class DatastoreDataset(Dataset):
    def __init__(self, datastore):
        self.store = datastore
        
    def __getitem__(self, key: int):
        items = self.store[key]
        
        items = [torch.from_numpy(np.copy(item)) if isinstance(item, np.ndarray) else item for item in items]
        
        if items[3].dtype == torch.float32:
            items[3] *= 255.
            items[3] = items[3].long()
        
        # Temp fix
        if items[3].dtype != torch.long:
            items[3] = items[3].long()
       #     items[3] = items[3].float() / 255.
        
        return items
        
    def __len__(self):
        return self.store.size
    
    
        