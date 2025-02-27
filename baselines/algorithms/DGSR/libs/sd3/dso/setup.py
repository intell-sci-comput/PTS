from distutils.core import setup
import os
from setuptools import dist

dist.Distribution().fetch_build_eggs(['Cython', 'numpy'])

import numpy
from Cython.Build import cythonize

required = [
    "pytest",
    "cython",
    "numpy",
    "numba",
    "sympy",
    "pandas",
    "scikit-learn",
    "click",
    "deap",
    "pathos",
    "seaborn",
    "progress",
    "tqdm",
    "commentjson",
    "PyYAML"
]

extras = {
    "control": [
        "mpi4py",
        "gym[box2d]",
        "pybullet",
        "stable-baselines[mpi]"
    ],
    "regression": []
}
extras['all'] = list(set([item for group in extras.values()
                     for item in group]))

setup(name='dso',
      version='1.0dev',
      description='Deep symbolic optimization.',
      author='LLNL',
      packages=['dso'],
      setup_requires=["numpy", "Cython"],
      ext_modules=cythonize([os.path.join('dso', 'cyfunc.pyx')]),
      include_dirs=[numpy.get_include()],
      install_requires=required,
      extras_require=extras
      )
