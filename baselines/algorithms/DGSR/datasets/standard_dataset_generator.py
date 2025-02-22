import random

lines = []
# dims_to_gen = list(range(4,100))
# dims_to_gen = list([2**pow for pow in range(2,10)])
# dims_to_gen = list([2*l for l in range(3,20)])
dims_to_gen = range(5, 10)

operators_to_sample_with = ["+", "-", "*", "/"]

# Associative
for dim in dims_to_gen:
    vars = [f"x{i+1}" for i in range(dim)]  # pylint: disable=redefined-builtin
    f_str = ""
    for i in range(len(vars)):
        f_str += vars[i] + random.choice(operators_to_sample_with)
    f_str = f_str[:-1]
    name = f"l_cd_{dim}"
    samples = dim * 100
    function_set = "A1"
    line = (
        name
        + ","
        + str(dim)
        + ","
        + '"{}"'.format(f_str)
        + ","
        + '"{{""all"":{{""U"":[-2,2,{}]}}}}"'.format(samples)
        + ","
        + "None"
        + ","
        + function_set
    )
    lines.append(line)

with open("tmp_out_ad.txt", "w") as fh:
    fh.write("\n".join(lines))
