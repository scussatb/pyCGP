from cgpes import CGPES
from cgp import CGP
import numpy as np

def evaluator(cgp, it):
	fit = 0.0
	x = 0.0
	while x < 2 * np.pi:
		fit += abs(np.sin(x) - cgp.run([x])[0])
		x += 0.1
	return -fit

def add(args):
	return args[0] + args[1]

def sub(args):
	return args[0] - args[1]

def mult(args):
	return args[0] * args[1]

def div(args):
	if args[1] != 0:
		return args[0] / args[1]
	else:
		return args[0]

def sin(args):
	return np.sin(args[0])

def cos(args):
	return np.cos(args[0])

def main():
	library = [CGP.CGPFunc(add, "add", 2), 
				CGP.CGPFunc(sub, "sub", 2), 
				CGP.CGPFunc(mult, "mult", 2), 
				CGP.CGPFunc(div, "div", 2)] 
	cgpFather = CGP.random(1, 1, 10, 1, library, 2)
	es = CGPES(4, 0.1, cgpFather, evaluator)
	es.run(1000000)

if __name__ == "__main__":
	main()
