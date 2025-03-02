import os
import random
from dso import DeepSymbolicOptimizer

# Create and train the model
dirs = os.listdir("../../../data/transformed")
with open("deep-symbolic-optimization/config_regression_gp.json") as f:
    raw_json = f.read()
for csv in dirs:
    json_data = raw_json.replace("data/transformed/DequanLi_xdot_3.csv","../../../data/transformed/"+csv)
    json_data = json_data.replace("seed_replace", csv.split('.')[-2].split("_")[-1])
    with open("config.json",'w')as f2:
        f2.write(json_data)
    model = DeepSymbolicOptimizer("config.json")
    model.train()
        