import numpy as np

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
