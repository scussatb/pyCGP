import os

import sys

import gym
import gym_jsbsim

from main import *

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

#load(sys.argv[1], sys.argv[2])

#displayFunctions(sys.argv[2])

#toDot(sys.argv[2], sys.argv[2])


#quit()

def testGenome(c):
	testVal = True
	for i in range(len(c.genome) - c.num_outputs):
		if i % (c.max_arity + 1) == 0:
			#func
			if c.genome[i] > len(library):
				print('test failed at ' + str(i) + ' with value ' + str(c.genome[i]))
				testVal = False
		else:
			if c.genome[i] >= c.num_inputs + int(i / (c.max_arity + 1)):
				print('test failed at ' + str(i) + ' with value ' + str(c.genome[i]))
				testVal = False
	for i in range(c.num_outputs):
		if c.genome[c.num_cols * c.num_rows * (c.max_arity + 1) + i] >= c.num_cols * c.num_rows + c.num_inputs:
			print('test failed at output ' + str(i) + ' with value ' + str(c.genome[c.num_cols * c.num_rows * (c.max_arity + 1) + i])) 
			testVal = False
	return testVal

library = build_funcLib()
nb_tests = 10000
nb_chars = 75
message = 'Random genome generation test'
nb_points = nb_chars - len(message)
print(message, end='', flush=True)
for cpt in range(nb_tests):
	globalVal = True
	c = CGP.random(7, 4, 100, 1, library, 1.0)
	testVal = testGenome(c)
	if cpt % (nb_tests / nb_points) < 1:
		print('.', end='', flush=True)
	if not testVal:
		print('[Failed]')
		globalVal = False
		print('Test ' + str(cpt) + ': ' + str(testVal))
		print('Genome ' + str(c.genome))
if globalVal:
	print ('[Passed]')

message = '1-Random mutation test'
nb_points = nb_chars - len(message)
print(message, end='', flush=True)
for cpt in range(nb_tests):
	globalVal = True
	f = CGP.random(7, 4, 100, 1, library, 1.0)
	c = f.clone()
	c.mutate(1)
	testVal = testGenome(c)
	if cpt % (nb_tests / nb_points) < 1:
		print('.', end='', flush=True)
	if not testVal:
		print('[Failed]')
		globalVal = False
		print('Test ' + str(cpt) + ': ' + str(testVal))
		print('Father ' + str(f.genome))
		print('Offspring ' + str(c.genome))
if globalVal:
	print ('[Passed]')

message = 'Per gene mutation test'
nb_points = nb_chars - len(message)
print(message, end='', flush=True)
for cpt in range(nb_tests):
	globalVal = True
	f = CGP.random(7, 4, 100, 1, library, 1.0)
	c = f.clone()
	c.mutate_per_gene(0.1, 0.3)
	testVal = testGenome(c)
	if cpt % (nb_tests / nb_points) < 1:
		print('.', end='', flush=True)
	if not testVal:
		print('[Failed]')
		globalVal = False
		print('Test ' + str(cpt) + ': ' + str(testVal))
		print('Father ' + str(f.genome))
		print('Offspring ' + str(c.genome))
if globalVal:
	print ('[Passed]')

message = 'Goldman mutation test'
nb_points = nb_chars - len(message)
print(message, end='', flush=True)
for cpt in range(nb_tests):
	globalVal = True
	f = CGP.random(7, 4, 100, 1, library, 1.0)
	c = f.clone()
	c.goldman_mutate()
	testVal = testGenome(c)
	if cpt % (nb_tests / nb_points) < 1:
		print('.', end='', flush=True)
	if not testVal:
		print('[Failed]')
		globalVal = False
		print('Test ' + str(cpt) + ': ' + str(testVal))
		print('Father ' + str(f.genome))
		print('Offspring ' + str(c.genome))
if globalVal:
	print ('[Passed]')
	
