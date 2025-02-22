# Create a virtual environment with Python 3.9 (here using conda):
conda create --name dgsr python=3.9
conda activate dgsr

# Set up key packaging-related tools:
pip install --upgrade pip
pip install "setuptools<58.0.0"  # Required for installing deap==1.3.0

# Install dependencies:
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# ^ Or use the pip alternative torch installation command from https://pytorch.org/get-started/locally/
# Choose a different version of CUDA or CPU-only, as needed.
pip install -r requirements.txt
# pip install sympy==1.7.1 # Works better with sympy==1.10.1 - however cannot then generate libraries etc.

# Install third party library sd3/dso:
export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS"
# ^ Needed on Mac to prevent fatal error: 'numpy/arrayobject.h' file not found.
pip install -e ./libs/sd3/dso # Install DSO package and core dependencies
cd ./libs/sd3/dso
python setup.py build_ext --inplace
cd ../../..

# Install third party library NeuralSymbolicRegressionThatScales
cd ./libs/NeuralSymbolicRegressionThatScales
pip install -e ./src/
cd ../..

# Optional. For library development, install developement dependencies.
pip install -r requirements-dev.txt
