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

cgpFather = CGP.random(1, 1, 10, 1, 2)
es = CGPES(4, 0.1, cgpFather, evaluator)
es.run(1000000)
