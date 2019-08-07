from cgpes import CGPES
from cgp import CGP
from cgpfunctions import *
import numpy as np
from evaluator import Evaluator
import sys

class SinEvaluator(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, cgp, it, displayTrace = False):
        fit = 0.0
        x = 0.0
        while x < 1.0:
            cgpVal =  cgp.run([x])[0]
            toFit = np.sin(x)
            fit += abs(toFit - cgpVal)
            if displayTrace:
                print(x, '\t', toFit, '\t', cgpVal)
            x += 0.01
        return -fit

    def clone(self):
        return SinEvaluator()

def build_funcLib():
    return [CGP.CGPFunc(f_sum, 'sum', 2),
            CGP.CGPFunc(f_aminus, 'aminus', 2),
            CGP.CGPFunc(f_mult, 'mult', 2),
            CGP.CGPFunc(f_exp, 'exp', 2),
            CGP.CGPFunc(f_abs, 'abs', 1),
            CGP.CGPFunc(f_sqrt, 'sqrt', 1),
            CGP.CGPFunc(f_sqrtxy, 'sqrtxy', 2),
            CGP.CGPFunc(f_squared, 'squared', 1),
            CGP.CGPFunc(f_pow, 'pow', 2),
            CGP.CGPFunc(f_one, 'one', 0),
            CGP.CGPFunc(f_zero, 'zero', 0),
            CGP.CGPFunc(f_inv, 'inv', 1),
            CGP.CGPFunc(f_gt, 'gt', 2),
            CGP.CGPFunc(f_asin, 'asin', 1),
            CGP.CGPFunc(f_acos, 'acos', 1),
            CGP.CGPFunc(f_atan, 'atan', 1),
            CGP.CGPFunc(f_min, 'min', 2),
            CGP.CGPFunc(f_max, 'max', 2),
            CGP.CGPFunc(f_round, 'round', 1),
            CGP.CGPFunc(f_floor, 'floor', 1),
            CGP.CGPFunc(f_ceil, 'ceil', 1)
            ]


def evolveSin(folder_name, col=100, row=1, nb_ind=4, mutation_rate_nodes=0.1, mutation_rate_outputs=0.3,
              n_cpus=1, n_it=1000000, genome=None):
    library = build_funcLib()
    e = SinEvaluator()
    if genome is None:
        cgpFather = CGP.random(1, 1, col, row, library, 1.0)
    else:
        cgpFather = CGP.load_from_file(genome, library)
    print(cgpFather.genome)
    es = CGPES(nb_ind, mutation_rate_nodes, mutation_rate_outputs, cgpFather, e, folder_name, n_cpus)
    es.run(n_it)


def load(file_name):
    print('loading ' + file_name)
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    e = SinEvaluator()
    print(e.evaluate(c, 0, displayTrace=True))
   
def toDot(file_name, out_name):
    print('Exporting ' + file_name + ' in dot ' + out_name + '.dot')
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    c.to_dot(out_name+'.dot', ['x'], ['y'])
    print('Converting dot file into pdf in ' + out_name + '.pdf')
    os.system('dot -Tpdf ' + out_name + '.dot' + ' -o ' + out_name + '.pdf')

def displayFunctions(file_name):
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    c.to_function_string(['x'], ['y'])


if __name__ == '__main__':
    print (sys.argv)
    if len(sys.argv)==2:
        print('Starting evolution from random genome')
        evolveSin(sys.argv[1])
    elif len(sys.argv)==3:
        print('Starting evolution from genome saved in ', sys.argv[2])
        evo(sys.argv[1], genome=sys.argv[2])