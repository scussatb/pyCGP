import os
os.environ['JSBSIM_ROOT_DIR'] = '/home/cussat/Documents/Airbus_CGP/JSBSim/jsbsim-JSBSim-trusty-v2018a'

import sys
from cgpes import CGPES
from cgp import CGP
from cgpfunctions import *
import numpy as np
from evaluator import Evaluator
import gym
import gym_jsbsim
import sys
import matplotlib.pyplot as plt



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

	def evaluate(self, cgp, it, with_render=False, display_in_out=False, plot=False):
		if plot:
			fig = plt.figure()
			ax = fig.add_subplot(221, projection='3d')	
			ay = fig.add_subplot(222)
			az = fig.add_subplot(224)
			traceTraj = np.zeros((10000, 3))
			traceState = np.zeros((10000, len(self.env.observation_space.sample())))
			traceAction = np.zeros((10000, len(self.env.action_space.sample())))
		fitnesses = np.zeros(self.ep_max)
		#print(cgp.genome)
		for e in range(self.ep_max):
			end = False
			fit = 0
			states = self.env.reset()
			step = 0
			while not end:
				#print(states)
				for s in range(len(states)):
					states[s] = min(self.env.observation_space.high[s], max(self.env.observation_space.low[s], states[s]))
					states[s] = 2.0 * (states[s] - self.env.observation_space.low[s]) / (self.env.observation_space.high[s] - self.env.observation_space.low[s]) - 1.0
				#states = 2.0 * (states - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low) - 1.0
				#print('normalized: ', states)
				if plot:
					traceState[step] = states
				actions = cgp.run(states)
				for a in range(len(actions)):
					#outputs = outputs * 0.25
					actions[a] = np.minimum(np.maximum(actions[a], self.env.action_space.low[a]), self.env.action_space.high[a])
				if plot:
					traceAction[step] = actions
				if display_in_out:
					print(' ', states, ' => ', actions)
				states, reward, end, _ = self.env.step(actions)
				if with_render:
					#self.env.render()
					#print(self.env._get_full_state())
					full_state = self.env._get_full_state()
					print(step*0.2, ',', self.env._get_full_state()['position/long-gc-deg'], ',', self.env._get_full_state()['position/lat-geod-deg'], ',', self.env._get_full_state()['position/h-sl-ft'], ',')
				if plot:
					traceTraj[step] = [self.env._get_full_state()['position/long-gc-deg'], self.env._get_full_state()['position/lat-geod-deg'], self.env._get_full_state()['position/h-sl-ft']]
					if step%1000 == 0:
						ax.scatter3D(self.env._get_full_state()['position/long-gc-deg'], self.env._get_full_state()['position/lat-geod-deg'], self.env._get_full_state()['position/h-sl-ft'])
						ay.scatter(step, 0)
						az.scatter(step, 0)
				#fit += reward * reward
				if abs(states[0]) < 0.05:
					fit += 0.1 * (1.0 - abs(states[0]))
				if abs(states[1]) < 0.2:
					fit += 1.0 - abs(states[1])
				#	fit += reward
				#fit += 1
				step += 1
			fitnesses[e] = fit	
		np.sort(fitnesses)
		fit = 0
		sum_e = 0
		for e in range(self.ep_max):
			fit += fitnesses[e] * (e + 1)
			sum_e += e + 1
		#print(fitnesses)
		#print(fit)
		#print('-----------')

		if plot:
			ay.plot(range(0, step), traceState[0:step, 0], label='delta_alt')
			ay.plot(range(0, step), traceState[0:step, 1], label='delat_head')
			ay.legend()
			az.plot(range(0, step), traceAction[0:step, 0], label='aileron')
			az.plot(range(0, step), traceAction[0:step, 1], label='elevator')
			az.plot(range(0, step), traceAction[0:step, 2], label='rudder')
			az.plot(range(0, step), traceAction[0:step, 3], label='throttle')
			az.legend()
			
			ax.plot3D(traceTraj[0:step,0], traceTraj[0:step,1], traceTraj[0:step,2], 'black')
			plt.show()
		return fit / sum_e 
	
	def clone(self):
		return GymEvaluator(self.env_name, self.it_max, self.ep_max)

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

def evolveGym(env, library, folder_name, col=100, row=1, nb_ind=4, mutation_rate=0.1, n_cpus=1, n_it=1000000):
	e = GymEvaluator(env, 10000, 3)
	#cgpFather = CGP.random(len(e.env.observation_space.sample()), len(e.env.action_space.sample()), col, row, library)
	print(e.env.observation_space.sample())
	print(e.env.action_space.sample())
	cgpFather = CGP.random(len(e.env.observation_space.sample()), len(e.env.action_space.sample()), col, row, library)
	print(cgpFather.genome)
	es = CGPES(nb_ind, mutation_rate, cgpFather, e, folder_name, n_cpus)
	es.run(n_it)
	

def evo(gym_env, folder_name):
	library = build_funcLib() 
	#evolveGym('JSBSim-ChangeHeadingControlTask-A320-NoFG-v0', library)
	evolveGym(gym_env, library, folder_name)

def load(file_name):
	print ('loading '+file_name)
	library = build_funcLib()
	c = CGP.load_from_file(file_name, library)
	e = GymEvaluator('JSBSim-ChangeHeadingControlTask-A320-NoFG-v0', 10000, 1)
	#e = GymEvaluator('JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0', 10000, 5)
	print(e.evaluate(c, 0, True, False, True))

def toDot(file_name, dot_name):
	print('Exporting ' + file_name + ' in dot ' + dot_name)
	library = build_funcLib()
	c = CGP.load_from_file(file_name, library)
	c.to_dot(dot_name)

if __name__ == '__main__':
	evo(sys.argv[1], sys.argv[2])
