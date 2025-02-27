import sys
from random import random
from subprocess import call
from pprint import pprint
sys.path.append('..')
from mcmc import Tree

def read_trace(inFileName):
    with open(inFileName) as inf:
        lines = inf.readlines()
        try:
            variables = eval(lines[2].split('Variables:  ')[1])
        except:
            variables = eval(lines[1].split('Variables:  ')[1])
    with open(inFileName) as inf:
        tracedict = dict([(line.strip().split(' || ')[6],
                           line.strip().split(' || '))
                          for line in inf.readlines()
                          if not line.startswith('#') and
                          ' || ' in line])
        trace = [l for c, l in tracedict.items()]
    return trace, variables

def get_top(trace, n=10, col=1, reverse=False):
    dec_trace = [(float(line[col]), random(), line) for line in trace]
    dec_trace.sort()
    if reverse:
        dec_trace.reverse()
    return [line for s, tb, line in dec_trace[:n]]

def parse_model_properties(inFileName='.model.tmp'):
    data = {}
    with open(inFileName) as inf:
        for line in [l for l in inf.readlines() if not l.startswith('--')]:
            k = line.strip().split()[0][:-1]
            v = ' '.join(line.strip().split()[1:])
            data[k] = v
    return data

def main(inFileName):
    dset = inFileName.split('/')[0]
    priorFileName = inFileName[inFileName.find('__')+2 :
                               inFileName.find('.dat')+4]

    trace, variables = read_trace(inFileName)
    top = get_top(trace, col=1, n=50)
    model_prop = {}
    for i in range(len(top)):
        print '>>>>> MODEL', i+1, top[i][1], top[i][3], top[i][6]
        cm = 'python model_charac.py -l -p \"../Prior/%s\" -v \"%s\" %s \"%s\"' % (
            priorFileName,
            top[i][8],
            dset,
            top[i][7],
        )
        call(cm + '> .model.tmp', shell=True)
        model_prop[i] = parse_model_properties()
        call('cat .model.tmp', shell=True)

    # Print the LaTeX table
    var2x = {}
    for i in range(len(variables)):
        tt = Tree(from_string=str(variables[i]))
        var2x[tt.latex()] = 'x_{%d}' % i
    print '\\hline'
    print 'Variables\\\\'
    print '\\hline'
    print '\\\\\n'.join(['\\multicolumn{4}{l}{$%s$: $%s$}' % (x, v) for v, x in var2x.items()]) + '\\\\'
    print '\\hline'
    print 'Model, $f$ & $\log p(f|D)$ & $B(f)$ & MAE\\\\'
    print '\\hline'

    for i in range(len(model_prop)):
        # Start replacing the longest vars to avoid replacing substrings (e.g. GPD in GPDRD)
        sorted_vars = [(len(v), v) for v in var2x]
        sorted_vars.sort()
        sorted_vars.reverse()
        try:
            for l, v in sorted_vars:
                model_prop[i]['LaTeX'] = model_prop[i]['LaTeX'].replace(v, var2x[v])
            print ' & '.join((
                '$%s$' % model_prop[i]['LaTeX'],
                '%.1f' % -float(model_prop[i]['-log(F)'].split(', ')[0][1:]),
                '%.1f' % float(model_prop[i]['BIC']),
                model_prop[i]['LOO-MAE'].replace('+-', ' $\\pm $'),
            )) + ' \\\\'
        except KeyError:
            pass

        
if __name__ == '__main__':
    inFileName = sys.argv[1]
    main(inFileName)
