import pandas as pd
import click
import sympy as sp

@click.command()
@click.option("--data_path", default="data/csv_old/ours_all_flavours.csv")
def main(data_path):
    csv = pd.read_csv(data_path)
    remap = {"x":"x_1","y":"x_2","z":"x_3"}
    fin_row = []
    for i in range(len(csv)):
        curr_row = {}
        sym = sp.sympify(csv["gt_expr"].loc[i])
        sym = sym.subs(sp.Symbol("x"), sp.Symbol(remap["x"]))
        sym = sym.subs(sp.Symbol("y"), sp.Symbol(remap["y"]))
        sym = sym.subs(sp.Symbol("z"), sp.Symbol(remap["z"]))
        curr_row["eq"] = str(sym)
        support_dict = eval(csv["support"].loc[i])
        new_support_dict = {}
        for key in support_dict.keys():
            new_support_dict[remap[key]] = {"max": support_dict[key]["U"][1], "min": support_dict[key]["U"][0]}
        curr_row["support"] = new_support_dict
        num_points = 500
        curr_row["num_points"] = num_points
        fin_row.append(curr_row)
    df = pd.DataFrame(fin_row)
    df.to_csv("data/benchmark/old_test.csv")

if __name__=="__main__":
    main()