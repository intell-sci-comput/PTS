# NeuralSymbolicRegressionThatScales

Pytorch implementation and pretrained models for the paper "Neural Symbolic Regression That Scales", presented at ICML 2021. 
Our deep-learning based approach is the first symbolic regression method that leverages large scale pre-training. We procedurally generate an unbounded set of equations, and simultaneously pre-train a Transformer to predict the symbolic equation from a corresponding set of input-output-pairs. 

For details, see **Neural Symbolic Regression That Scales**.  [[`arXiv`](https://arxiv.org/pdf/2106.06427.pdf)] 


## Installation
Please clone and install this repository via

```
git clone https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales.git
cd NeuralSymbolicRegressionThatScales/
pip3 install -e src/
```

This library requires python>3.7



## Pretrained models
We offer two models, "10M" and "100M".  Both are trained with parameter configuration showed in **dataset_configuration.json** (which contains details about how datasets are created) and **scripts/config.yaml** (which contains details of how models are trained). "10M" model is trained with 10 million datasets and "100M" model is trained with 100 millions dataset.

* Link to 100M: [[Link](https://drive.google.com/drive/folders/1LTKUX-KhoUbW-WOx-ZJ8KitxK7Nov41G?usp=sharing)]
* Link to 10M: [[Link](https://drive.google.com/file/d/1cNZq3dLnSUKEm-ujDl2mb2cCCorv7kOC/view?usp=sharing)]

If you want to try the models out, look at **jupyter/fit_func.ipynb**. Before running the notebook, make sure to first create a folder named "weights" and to download the provided checkpoints there.


## Dataset Generation
Before training, you need a dataset of equations. Here the steps to follow

### Raw training dataset generation
The equation generator scripts are based on [[SymbolicMathematics](https://github.com/facebookresearch/SymbolicMathematics)]
First, if you want to change the defaults value, configure the dataset_configuration.json file:
```
{
    "max_len": 20, #Maximum length of an equation
    "operators": "add:10,mul:10,sub:5,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:2", #Operator unnormalized probability
    "max_ops": 5, #Maximum number of operations
    "rewrite_functions": "", #Not used, leave it empty
    "variables": ["x_1","x_2","x_3"], #Variable names, if you want to add more add follow the convention i.e. x_4, x_5,... and so on
    "eos_index": 1,
    "pad_index": 0
}
```
There are two ways to generate this dataset:

* If you are running on linux, you use makefile in terminal as follows:
```
export NUM=${NumberOfEquations} #Export num of equations
make data/raw_datasets/${NUM}: #Launch make file command
```
NumberOfEquations can be defined in two formats with K or M suffix. For instance 100K is equal to 100'000 while 10M is equal to 10'0000000
For example, if you want to create a 10M dataset simply:

```
export NUM=10M #Export num variable
make data/raw_datasets/10M: #Launch make file command
```

* Run this script: 
```
python3 scripts/data_creation/dataset_creation.py --number_of_equations NumberOfEquations --no-debug #Replace NumberOfEquations with the number of equations you want to generate
```

After this command you will have a folder named **data/raw_data/NumberOfEquations** containing .h5 files. By default, each of this h5 files contains a maximum of 5e4 equations. 


### Raw test dataset generation
This step is optional. You can skip it if you want to use our test set used for the paper (located in **test_set/nc.csv**).
Use the same commands as before for generating a validation dataset. All equations in this dataset will be remove from the training dataset in the next stage, 
hence this validation dataset should be **small**. For our paper it constisted of 200 equations.

```
#Code for generating a 150 equation dataset 
python3 scripts/data_creation/dataset_creation.py --number_of_equations 150 --no-debug #This code creates a new folder data/raw_datasets/150
```

If you want, you can convert the newly created validation dataset in a csv format. 
To do so, run: `python3 scripts/csv_handling/dataload_format_to_csv.py raw_test_path=data/raw_datasets/150`
This command will create two csv files named test_nc.csv (equations without constants) and test_wc.csv (equation with constants) in the test_set folder.

### Remove test and numerical problematic equations from the training dataset 
The following steps will remove the validation equations from the training set and remove equations that are always nan, inf, etc.
* `path_to_data_folder=data/raw_datasets/100000`  if you have created a 100K dataset
* `path_to_csv=test_set/test_nc.csv` if you have created 150 equations for validation. If you want to use the one in the paper replace it with `nc.csv`
```
python3 scripts/data_creation/filter_from_already_existing.py --data_path path_to_data_folder --csv_path path_to_csv #You can leave csv_path empty if you do not want to create a validation set
python3 scripts/data_creation/apply_filtering.py --data_path path_to_data_folder 
```
You should now have a folder named data/datasets/100000. This will be the training folder.

## Training
Once you have created your training and validation datasets run 
```
python3 scripts/train.py
```
You can configure the config.yaml with the necessary options. Most important, make sure you have set 
train_path and val_path correctly. If you have followed the 100K example this should be set as:
```
train_path:  data/datasets/100000
val_path: data/raw_datasets/150
```

