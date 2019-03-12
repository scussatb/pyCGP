from cgpes import CGPES
from cgp import CGP
from cgpfunctions import *
import numpy as np
from evaluator import Evaluator
import gym
import sys

class SinEvaluator(Evaluator):
	def __init__(self):
		super().__init__()

	def evaluate(self, cgp, it):
		fit = 0.0
		x = 0.0
		while x < 2 * np.pi:
			fit += abs(np.sin(x) - cgp.run([x])[0])
			x += 0.1
		return -fit
	
	def clone(self):
		return SinEvaluator()

class GymEvaluator(Evaluator):
	def __init__(self, env_name, it_max, ep_max):
		super().__init__()
		self.env_name = env_name
		self.it_max = it_max
		self.ep_max = ep_max
		self.env = gym.make(self.env_name)

	def evaluate(self, cgp, it, with_render=False, display_in_out=False):
		fit_min = sys.float_info.max
		fit_max = sys.float_info.min
		fit_sum = 0
		for e in range(self.ep_max):
			fit = 0
			inputs = self.env.reset()
			#print(cgp.genome)
			for i in range(self.it_max):
				outputs = cgp.run(inputs)
				if display_in_out:
					print(i, ' ', inputs, ' => ', outputs)
				if with_render:
					self.env.render()
				inputs, reward, end, _ = self.env.step(outputs)
				fit += reward
			fit_min = np.minimum(fit_min, fit)
			fit_max = np.maximum(fit_max, fit)
			fit_sum += fit
		return fit_sum / self.ep_max
		#return (fit_min + fit_max) / 2.0
		#return fit_min
	
	def clone(self):
		return GymEvaluator(self.env_name, self.it_max, self.ep_max)

def main():
	library = [CGP.CGPFunc(add, 'add', 2), 
				CGP.CGPFunc(sub, 'sub', 2), 
				CGP.CGPFunc(mult, 'mult', 2), 
				CGP.CGPFunc(div, 'div', 2),
				CGP.CGPFunc(sin, 'sin', 1),
				CGP.CGPFunc(cos, 'cos', 1)] 
	# sin test
	#cgpFather = CGP.random(1, 1, 10, 1, library, 2)
	#e = SinEvaluator()
	
	# gym lunar test
	cgpFather = CGP.random(8, 2, 10, 1, library, 2)
	e = GymEvaluator('LunarLanderContinuous-v2', 500, 3)
	es = CGPES(4, 0.1, cgpFather, e, 4)
	es.run(20)

def load(file_name):
	library = [CGP.CGPFunc(add, 'add', 2), 
				CGP.CGPFunc(sub, 'sub', 2), 
				CGP.CGPFunc(mult, 'mult', 2), 
				CGP.CGPFunc(div, 'div', 2), 
				CGP.CGPFunc(sin, 'sin', 1),
				CGP.CGPFunc(cos, 'cos', 1)] 
	c = CGP.load_from_file(file_name, library, 2)
	e = GymEvaluator('LunarLanderContinuous-v2', 500, 100)
	print(e.evaluate(c, 0, True, True))

if __name__ == '__main__':
	main()
