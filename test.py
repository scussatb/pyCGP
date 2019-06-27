import os
os.environ['JSBSIM_ROOT_DIR'] = '/home/cussat/Documents/Airbus_CGP/JSBSim/jsbsim-JSBSim-trusty-v2018a'

import sys

import gym
import gym_jsbsim

from main import load

#print('make')
#env = gym.make('JSBSim-ChangeHeadingControlTask-A320-NoFG-v0')
#print('reset')
#env.reset()
#print(env.action_space.low)
#print(env.action_space.high)
#d = False
#while not d:
#	print('step')
#	s, e, d, _ = env.step(env.action_space.sample())
#	print(s)
#	print(e)
#	print(d)
#print(env._get_full_state())


load(sys.argv[1])
