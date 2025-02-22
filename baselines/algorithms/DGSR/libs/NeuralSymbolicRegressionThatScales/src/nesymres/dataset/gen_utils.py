def remap_independent_vars_to_monotic(f_expr):
    xs = [i for i in f_expr if 'x' == i[0]]
    xss = list(set(xs))
    xss.sort()
    to_map_to = [f'x_{j+1}' for j in range(len(xss))]
    return [to_map_to[xss.index(i)] if 'x' == i[0] else i for i in f_expr]

def test_remap_independent_vars_to_monotic():
    assert remap_independent_vars_to_monotic(['pow4', 'div', 'div', 'x_9', 'x_1', 'x_9']) == ['pow4', 'div', 'div', 'x_2', 'x_1', 'x_2']

def remap_to_include_all_vars(f_expr):
    f_exp = ''
    j = 1
    for f in f_expr:
        if 'x' == f[0]:
            f_exp += f'x_{j},'
            j+=1
        else:
            f_exp += f + ','
    return f_exp[:-1].split(',')

if __name__ == "__main__":
    test_remap_independent_vars_to_monotic()
    print('tests passed')

