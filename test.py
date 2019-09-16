from main import *

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
nb_tests = 1
nb_chars = 75
message = 'Random genome generation test'
nb_points = nb_chars - len(message)
print(message, end='', flush=True)
for cpt in range(nb_tests):
	globalVal = True
	c = CGP.random(7, 4, 100, 1, library, recurrency_distance=1.0, recursive=False)
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
nb_tests = 1
nb_points = nb_chars - len(message)
print(message, end='', flush=True)
for cpt in range(nb_tests):
	globalVal = True
	f = CGP.random(7, 4, 100, 1, library, recurrency_distance=1.0, recursive=False)
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
nb_tests = 100000
nb_points = nb_chars - len(message)
print(message, end='', flush=True)
for cpt in range(nb_tests):
	globalVal = True
	f = CGP.random(7, 4, 100, 1, library, recurrency_distance=1.0, recursive=False)
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
nb_tests = 10000
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
	

