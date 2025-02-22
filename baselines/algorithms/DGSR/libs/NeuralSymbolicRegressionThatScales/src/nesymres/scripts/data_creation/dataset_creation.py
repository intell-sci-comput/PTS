import numpy as np
# from torch import multiprocessing
import multiprocessing
# from torch.multiprocessing import Manager
import click
import warnings
from tqdm import tqdm
import json
import os
from nesymres.dataset import generator
import time
import signal
from nesymres import dclasses
from pathlib import Path
import pickle
from sympy import lambdify
from nesymres.utils import create_env, H5FilesCreator
from nesymres.utils import code_unpickler, code_pickler
import copyreg
import types
from itertools import chain
import traceback
import h5py
import warnings
import shutil
from dotenv import load_dotenv
from functools import partial

class Pipepile:
    def __init__(self, env: generator.Generator, number_of_equations, eq_per_block, h5_creator: H5FilesCreator, is_timer=False):
        self.env = env
        #manager = Manager()
        #self.cnt = manager.list()
        self.is_timer = is_timer
        self.number_of_equations = number_of_equations
        self.fun_args = ",".join(chain(list(env.variables), env.coefficients))
        self.eq_per_block = eq_per_block
        self.h5_creator = h5_creator
        self.errors = {}

    def create_block(self, block_idx, global_seed=0):
        block = []
        counter = block_idx
        hlimit = block_idx + self.eq_per_block
        t0 = time.time()
        prev_count = 0
        while counter < hlimit and counter < self.number_of_equations:
            res = self.return_training_set(counter, global_seed=global_seed)
            block.append(res)
            counter = counter + 1
            if counter % 1000 == 0:
                delta = counter - prev_count
                prev_count = counter
                eqs = delta / (time.time() - t0)
                t0 = time.time()
                print(f'[Block id {block_idx}] Time left on block {(hlimit - counter) / eqs}')
        self.h5_creator.create_single_hd5_from_eqs(
            (block_idx // self.eq_per_block, block))
        return self.errors

    def handler(self, signum, frame):
        raise TimeoutError

    def return_training_set(self, i, global_seed=0) -> dclasses.Equation:
        np.random.seed(i + global_seed)
        while True:
            # print(self.errors)
            try:
                res = self.create_lambda(np.random.randint(2**32 - 1))
                assert type(res) == dclasses.Equation
                return res
            except TimeoutError:
                signal.alarm(0)
                continue
            except generator.NotCorrectIndependentVariables as e:
                signal.alarm(0)
                if e.args[0] in self.errors:
                    self.errors[e.args[0]] += 1
                else:
                    self.errors[e.args[0]] = 1
                continue
            except generator.UnknownSymPyOperator:
                signal.alarm(0)
                continue
            except generator.ValueErrorExpression as e:
                signal.alarm(0)
                if e.args[0] in self.errors:
                    self.errors[e.args[0]] += 1
                else:
                    self.errors[e.args[0]] = 1
                continue
            except generator.ImAccomulationBounds:
                signal.alarm(0)
                continue
            except RecursionError:
                # Due to Sympy
                signal.alarm(0)
                continue
            except ValueError as e:
                signal.alarm(0)
                if e.args[0] in self.errors:
                    self.errors[e.args[0]] += 1
                else:
                    self.errors[e.args[0]] = 1
                continue
            except KeyError:
                signal.alarm(0)
                continue
            except TypeError:
                signal.alarm(0)
            # except Exception as E:
            #     continue

    def create_lambda(self, i):
        if self.is_timer:
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(60)
        prefix, variables = self.env.generate_equation(np.random)
        prefix = self.env.add_identifier_constants(prefix)
        consts = self.env.return_constants(prefix)
        infix, _ = self.env._prefix_to_infix(
            prefix, coefficients=self.env.coefficients, variables=self.env.variables)
        consts_elemns = {y: y for x in consts.values() for y in x}
        constants_expression = infix.format(**consts_elemns)
        eq = lambdify(
            self.fun_args,
            constants_expression,
            modules=["numpy"],
        )
        res = dclasses.Equation(expr=infix, code=eq.__code__,
                                coeff_dict=consts_elemns, variables=variables)  # Main Eq creation
        signal.alarm(0)
        return res


# @click.command()
# @click.option(
#     "--number_of_equations",
#     default=200,
#     help="Total number of equations to generate",
# )
# @click.option(
#     "--eq_per_block",
#     default=5e4,
#     help="Total number of equations to generate",
# )
# @click.option("--debug/--no-debug", default=True)
def creator(config="dataset_configuration.json", number_of_equations=200, eq_per_block=5e4, debug=False, ds_key='', test_dataset=False, global_seed=0):
    # Needed for serializing code objects
    copyreg.pickle(types.CodeType, code_pickler, code_unpickler)
    load_dotenv()
    CPU_COUNT_DIV = int(os.getenv('CPU_COUNT_DIV')) if os.getenv('CPU_COUNT_DIV') else 1
    cpus_available = multiprocessing.cpu_count() // CPU_COUNT_DIV
    eq_per_block = min(number_of_equations // cpus_available, int(eq_per_block))
    if eq_per_block == 0:
        eq_per_block = 1
        number_of_equations = cpus_available
        print(f'Setting equations to generate to minimum of {number_of_equations} equations, with eqs per block of {eq_per_block}')
    print("There are {} equations per block. The progress bar will have this resolution".format(
        eq_per_block))
    # warnings.filterwarnings("error")
    load_dotenv()
    path_pre_fix = os.getenv('DATA_DIR') if os.getenv('DATA_DIR') else ''
    env, param, config_dict = create_env(config)
    if not test_dataset:
        mid_path = 'raw_datasets'
    else:
        mid_path = 'test_datasets'
    if not debug:
        folder_path = Path(f"{path_pre_fix}data/{mid_path}/{ds_key}-{number_of_equations}")
    else:
        folder_path = Path(f"{path_pre_fix}data/{mid_path}/debug/{ds_key}-{number_of_equations}")
    print(f'Creating dataset: {folder_path}')
    # Remove folder for re-creating
    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)
    h5_creator = H5FilesCreator(target_path=folder_path)
    env_pip = Pipepile(env,
                       number_of_equations=number_of_equations,
                       eq_per_block=eq_per_block,
                       h5_creator=h5_creator,
                       is_timer=not debug)
    create_block_func = partial(env_pip.create_block, global_seed=global_seed)
    starttime = time.time()
    func = []
    res = []
    counter = []
    if not debug:
        try:
            with multiprocessing.Pool(1) as p:
                max_ = number_of_equations
                with tqdm(total=max_) as pbar:
                    for f in p.imap_unordered(
                        create_block_func, range(0, number_of_equations, eq_per_block)
                    ):
                        pbar.update(eq_per_block)
                        res.append(f)
        except:
            print(traceback.format_exc())

    else:
        list(map(create_block_func, tqdm(range(0, number_of_equations, eq_per_block))))

    all_dict = {}
    for r in res:
        if r:
            for k, v in r.items():
                if k in all_dict:
                    all_dict[k] += v
                else:
                    all_dict[k] = v

    print(f'Errors seen in generation: {all_dict}')
    dataset = dclasses.DatasetDetails(
        config=config_dict,
        total_coefficients=env.coefficients,
        total_variables=list(env.variables),
        word2id=env.word2id,
        id2word=env.id2word,
        una_ops=env.una_ops,
        bin_ops=env.una_ops,
        rewrite_functions=env.rewrite_functions,
        total_number_of_eqs=number_of_equations,
        eqs_per_hdf=eq_per_block,
        generator_details=param)
    print("Expression generation took {} seconds".format(time.time() - starttime))
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    t_hf = h5py.File(os.path.join(folder_path, "metadata.h5"), 'w')
    t_hf.create_dataset("other", data=np.void(pickle.dumps(dataset)))
    t_hf.close()
    return folder_path


if __name__ == "__main__":
    creator()
