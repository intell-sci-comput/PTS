import numpy as np
import pandas as pd

# feynman_dataset = './datasets/FeynmanEquations.csv'
feynman_dataset = "./datasets/BonusEquations.csv"

df = pd.read_csv(feynman_dataset)
df = df.sort_values(by=["# variables"], ascending=False)
df = df.reset_index()

# dim = 3
name = "all"
# sample_points = 10 * dim

dl = []
for index, row in df.iterrows():
    # if index >= 60:
    #     print('as')
    #     pass
    f = row["Formula"]
    if type(f) != str:
        continue
    f = f.replace("sqrt", "$")
    f = f.replace("exp", "&")
    f = f.replace("sin", "€")
    f = f.replace("cos", "£")
    f = f.replace("tan", "%")
    f = f.replace("ln(", "±")
    for i in range(10):
        v = row[f"v{i+1}_name"]
        if type(v) == str:
            f = f.replace(v, f"@{i+1}")
    f = f.replace("@", "x")
    f = f.replace("$", "sqrt")
    f = f.replace("&", "exp")
    f = f.replace("€", "sin")
    f = f.replace("£", "cos")
    f = f.replace("%", "tan")
    f = f.replace("±", "ln(")
    sample_points = int(row["# variables"]) * 10
    if type(row["v1_low"]) == float and not np.isnan(row["v1_low"]):
        support = (
            "{"
            "all"
            ":{"
            "U"
            ":[" + str(int(row["v1_low"])) + "," + str(int(row["v1_high"])) + "," + str(sample_points) + "]}}"
        )
        dl.append(
            {
                "name": f"fn-{index}",
                "variables": int(row["# variables"]),
                "expression": f"{f}",
                "train_spec": support,
                "test_spec": None,
                "function_set": "Koza",
            }
        )


dm = pd.DataFrame(dl)

# Sort
dm = dm[dm["expression"].str.contains("|".join(["pi", "arc", "sqrt", "ega", "tanh", "epsil"])) == False]  # noqa: E712
dm = dm.drop_duplicates(subset=["expression"])
dm = dm[dm["variables"] >= 1]
dm = dm.reset_index()
dm["name"] = [f"fn_d_{name}_{i+1}" for i in dm.index]

# dm = dm.sample(n=10, random_state=0)
dm.to_csv("feynman_benchmark_new.csv", index=False, na_rep="None")
print(dm)
