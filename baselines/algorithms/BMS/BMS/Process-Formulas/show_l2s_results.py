import sys
import ast
# import numpy
# import seaborn
# import matplotlib

from utils import SYMPY_OP_STRING

num_vars = {}
num_unique_vars = {}
num_operations = {}
num_unique_operations = {}
operation_type = {}
operation_type_sq = {}

# operation name dictionary manu->roger
M2R = {
    'sinh' : 'sinh',
    'cos' : 'cos',
    'log' : 'log',
    'tanh' : 'tanh',
    'Pow2' : 'pow2',
    'Neg' : '-',
    'Abs' : 'abs',
    'sqrt' : 'sqrt',
    'cosh' : 'cosh',
    'factorial' : 'fac',
    'Add' : '+',
    'Pow' : '**',
    'exp' : 'exp',
    'Pow3' : 'pow3',
    'Mul' : '*',
    'Div' : '/',
    'sin' : 'sin',
    'tan' : 'tan',
    'sec' : 'sec',
}

if __name__ == "__main__":
    inFileName = sys.argv[1]

    inf = open(inFileName)
    num_vars = {}
    num_unique_vars = {}
    num_operations = {}
    num_unique_operations = {}

    for l in inf.readlines():
        data = ast.literal_eval(l.split('#')[1].strip())

        n_var = data['Vars']['num']
        n_uvar = data['Vars']['uniqnum']
        n_ops = data['Ops']['num']
        n_uops = data['Ops']['uniqnum']
        n_op_type = data['Ops']['ops']

        if n_var in num_vars:
            num_vars[n_var] += 1
        else:
            num_vars[n_var] = 1

        if n_uvar in num_unique_vars:
            num_unique_vars[n_uvar] += 1
        else:
            num_unique_vars[n_uvar] = 1

        if n_ops in num_operations:
            num_operations[n_ops] += 1
        else:
            num_operations[n_ops] = 1

        if n_uops in num_unique_operations:
            num_unique_operations[n_uops] += 1
        else:
            num_unique_operations[n_uops] = 1

        for k, v in n_op_type.iteritems():
            if k in SYMPY_OP_STRING:
                k = SYMPY_OP_STRING[k]
            if k in operation_type:
                operation_type[k] += v
                operation_type_sq[k] += v*v
            else:
                operation_type[k] = v
                operation_type_sq[k] = v*v

        if n_ops != 0:
            print n_var, n_uvar, n_ops, n_uops

    # print "Num vars:",num_vars
    f = open('num_vars.dat', 'w')
    for k, v in num_vars.iteritems():
        f.write("%s %s\n" % (k, v))
    f.close()
    # X = numpy.arange(len(num_vars))
    # matplotlib.pyplot.bar(X, num_vars.values(), align='center', width=0.5)
    # matplotlib.pyplot.xticks(X, num_vars.keys())
    # matplotlib.pyplot.title("Num vars")
    # ymax = max(num_vars.values()) + 1
    # matplotlib.pyplot.ylim(0, ymax)
    # matplotlib.pyplot.show()

    # print "Num unique vars:",num_unique_vars
    f = open('%s__num_unique_vars.dat' % inFileName, 'w')
    for k, v in num_unique_vars.iteritems():
        f.write("%s %s\n" % (k, v))
    f.close()
    # X = numpy.arange(len(num_unique_vars))
    # matplotlib.pyplot.bar(X, num_unique_vars.values(), align='center', width=0.5)
    # matplotlib.pyplot.xticks(X, num_unique_vars.keys())
    # matplotlib.pyplot.title("Num unique vars")
    # ymax = max(num_unique_vars.values()) + 1
    # matplotlib.pyplot.ylim(0, ymax)
    # matplotlib.pyplot.show()

    # print "Num operations:",num_operations
    f = open('%s__num_operations.dat' % inFileName, 'w')
    for k, v in num_operations.iteritems():
        f.write("%s %s\n" % (k, v))
    f.close()
    # X = numpy.arange(len(num_operations))
    # matplotlib.pyplot.bar(X, num_operations.values(), align='center', width=0.5)
    # matplotlib.pyplot.xticks(X, num_operations.keys())
    # matplotlib.pyplot.title("Num operations")
    # ymax = max(num_operations.values()) + 1
    # matplotlib.pyplot.ylim(0, ymax)
    # matplotlib.pyplot.show()

    # print "Num unique operations:",num_unique_operations
    f = open('%s__num_unique_operations.dat' % inFileName, 'w')
    for k, v in num_unique_operations.iteritems():
        f.write("%s %s\n" % (k, v))
    f.close()
    # X = numpy.arange(len(num_unique_operations))
    # matplotlib.pyplot.bar(X, num_unique_operations.values(), align='center', width=0.5)
    # matplotlib.pyplot.xticks(X, num_unique_operations.keys())
    # matplotlib.pyplot.title("Num unique operations")
    # ymax = max(num_unique_operations.values()) + 1
    # matplotlib.pyplot.ylim(0, ymax)
    # matplotlib.pyplot.show()

    print "Operation types", operation_type
    f = open('%s__operation_type.dat' % inFileName, 'w')
    for k, v in operation_type.iteritems():
        f.write("%s %s\n" % (M2R[k], v))
    f.close()
    # X = numpy.arange(len(operation_type))
    # matplotlib.pyplot.bar(X, operation_type.values(), align='center', width=0.5)
    # matplotlib.pyplot.xticks(X, operation_type.keys())
    # matplotlib.pyplot.title("Ops dist")
    # ymax = max(operation_type.values()) + 1
    # matplotlib.pyplot.ylim(0, ymax)
    # matplotlib.pyplot.show()

    print "Operation types squared", operation_type
    f = open('%s__operation_type_sq.dat' % inFileName, 'w')
    for k, v in operation_type_sq.iteritems():
        f.write("%s %s\n" % (M2R[k], v))
    f.close()
