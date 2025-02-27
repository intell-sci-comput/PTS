import numpy as np
import multiprocessing
import click
import os
from nesymres.utils import load_metadata_hdf5
from pathlib import Path
from nesymres.utils import H5FilesCreator
from tqdm import tqdm
from nesymres.utils import code_unpickler, code_pickler
import h5py
import pickle

def create_hdf_files(metadata, keep_true, base_path: Path, target_path: Path, debug) -> None:
    num_eqs_per_set = int(1e5)
    n_datasets = int(len(keep_true) // num_eqs_per_set) + 1
    h5_creator = H5FilesCreator(base_path, target_path,metadata)
    counter = 0
    print("Diving sets")
    sets = [[keep_true[idx] for idx in range(i*num_eqs_per_set,min((i+1)*num_eqs_per_set,len(keep_true)))] for i in range(n_datasets)]
    if not debug:
        with multiprocessing.Pool(4) as p: #multiprocessing.cpu_count()) as p:
            max_ = n_datasets
            with tqdm(total=max_) as pbar:
                for f in p.imap_unordered(
                    h5_creator.recreate_single_hd5_from_idx, enumerate(sets)
                ):
                    pbar.update()
    else:
        t = map(h5_creator.recreate_single_hd5_from_idx, tqdm(enumerate(sets)))
        target = list(t)
    total_number_of_eqs = len(keep_true)
    metadata.eqs = []
    metadata.total_number_of_eqs = total_number_of_eqs
    metadata.eqs_per_hdf = num_eqs_per_set
    t_hf = h5py.File(os.path.join(target_path, "metadata.h5") , 'w')
    t_hf.create_dataset("other", data=np.void(pickle.dumps(metadata)))
    t_hf.close()
    return


@click.command()
@click.option("--data_path", default="data/raw_datasets/10000000/")
@click.option("--debug/--no-debug", default=False)
def main(data_path,debug):
    bool_cond = np.load(os.path.join(data_path,"filtered.npy"),allow_pickle=True)
    entries = [idx for idx, entry in bool_cond if entry]
    metatada = load_metadata_hdf5(data_path)
    data_path = Path(data_path)
    target_path = Path(data_path.parent.parent / Path("datasets") / data_path.stem)
    create_hdf_files(metatada, entries, data_path, target_path, debug)
    




if __name__=="__main__":
    main()