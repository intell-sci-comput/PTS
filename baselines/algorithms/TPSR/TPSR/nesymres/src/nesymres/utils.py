import marshal
import copyreg
import types
import pickle
import json
from .dataset import generator
from .dclasses import DatasetDetails, Equation, GeneratorDetails
from typing import List, Tuple
import h5py
import os
import numpy as np
from pathlib import Path

class H5FilesCreator():
    def __init__(self,base_path: Path = None, target_path: Path = None, metadata=None):
        # if int(path.stem) > 1e6:
        #     file_name = base_path.parent / "{}M".format(int(base_path.stem) // 1e6)
        # elif int(path.stem) > 1e3:
        #     file_name = base_path.parent / "{}K".format(int(base_path.stem) // 1e3)
        target_path.mkdir(mode=0o777, parents=True, exist_ok=True)
        self.target_path = target_path
        
        self.base_path = base_path
        self.metadata = metadata
        

    def create_single_hd5_from_eqs(self,block):
        name_file, eqs = block
        t_hf = h5py.File(os.path.join(self.target_path, str(name_file) + ".h5") , 'w')
        for i, eq in enumerate(eqs):            
            curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=curr)
        t_hf.close()
    
    def recreate_single_hd5_from_idx(self,block:Tuple):
        name_file, eq_idxs = block
        t_hf = h5py.File(os.path.join(self.target_path, str(name_file) + ".h5") , 'w')
        for i, eq_idx in enumerate(eq_idxs):            
            eq = load_eq_raw(self.base_path, eq_idx, self.metadata.eqs_per_hdf)
            #curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=eq)
        t_hf.close()


def code_unpickler(data):
    return marshal.loads(data)

def code_pickler(code):
    return code_unpickler, (marshal.dumps(code),)

def load_eq_raw(path_folder, idx, num_eqs_per_set) -> Equation:
    index_file = str(int(idx/num_eqs_per_set))
    f = h5py.File(os.path.join(path_folder,f"{index_file}.h5"), 'r')
    dataset_metadata = f[str(idx - int(index_file)*int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    f.close()
    return raw_metadata

def load_eq(path_folder, idx, num_eqs_per_set) -> Equation:
    index_file = str(int(idx/num_eqs_per_set))
    f = h5py.File(os.path.join(path_folder,f"{index_file}.h5"), 'r')
    dataset_metadata = f[str(idx - int(index_file)*int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    f.close()
    return metadata

def load_metadata_hdf5(path_folder: Path) -> DatasetDetails:
    f = h5py.File(os.path.join(path_folder,"metadata.h5"), 'r')
    dataset_metadata = f["other"]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    return metadata

def create_env(path)->Tuple[generator.Generator,GeneratorDetails]:
    with open(path) as f:
        d = json.load(f)
    param = GeneratorDetails(**d)
    env = generator.Generator(param)
    return env, param, d

