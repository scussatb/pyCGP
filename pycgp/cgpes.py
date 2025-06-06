import os
import numpy as np
from .cgp import CGP
from .evaluator import Evaluator
from joblib import Parallel, delayed

class CGPES:
	def __init__(self, num_offsprings, mutation_rate_nodes, mutation_rate_outputs, father, evaluator, folder='genomes', num_cpus = 1):
		self.num_offsprings = num_offsprings
		self.mutation_rate_nodes = mutation_rate_nodes
		self.mutation_rate_outputs = mutation_rate_outputs
		self.father = father
		#self.num_mutations = int(len(self.father.genome) * self.mutation_rate)
		self.evaluator = evaluator
		self.num_cpus = num_cpus
		self.folder = folder
		if self.num_cpus > 1:
			self.evaluator_pool = []
			for i in range(self.num_offsprings):
				self.evaluator_pool.append(self.evaluator.clone())
		self.initialized = False

	def initialize(self):
		if not os.path.isdir(self.folder):
			os.mkdir(self.folder)
		self.logfile = open(self.folder + '/out.txt', 'w')
		self.current_fitness = self.evaluator.evaluate(self.father, 0)
		self.father.save(self.folder + '/cgp_genome_0_' + str(self.current_fitness) + '.txt')
		self.offsprings = np.empty(self.num_offsprings, dtype=type(self.father))
		self.offspring_fitnesses = np.zeros(self.num_offsprings, dtype=float)
		self.initialized = True
		self.it = 0

	def run(self, num_iteration, print_every = -1):
		if not self.initialized:
			self.initialize()

		for it in range(num_iteration):
			self.it += 1
			#generate offsprings
			if self.num_cpus == 1:
				for i in range(0, self.num_offsprings):
					self.offsprings[i] = self.father.clone()
					#self.offsprings[i].mutate(self.num_mutations)
					self.offsprings[i].mutate_per_gene(self.mutation_rate_nodes, self.mutation_rate_outputs)
#					self.offsprings[i].goldman_mutate_2()
					self.offspring_fitnesses[i] = self.evaluator.evaluate(self.offsprings[i], self.it)
			else:
				for i in range(self.num_offsprings):
					self.offsprings[i] = self.father.clone()
					#self.offsprings[i].mutate(self.num_mutations)
					#self.offsprings[i].mutate_per_gene(self.mutation_rate_nodes, self.mutation_rate_outputs)
					self.offsprings[i].goldman_mutate()
				def offspring_eval_task(offspring_id):
					return self.evaluator_pool[offspring_id].evaluate(self.offsprings[offspring_id], self.it)
				self.offspring_fitnesses = Parallel(n_jobs = self.num_cpus)(delayed(offspring_eval_task)(i) for i in range(self.num_offsprings)) 
			#get the best fitness
			best_offspring = np.argmax(self.offspring_fitnesses)
			if not self.evaluator.is_cacheable(self.it):
				self.current_fitness = self.evaluator.evaluate(self.father, self.it)
			#compare to father
			self.father_was_updated = False
			if self.offspring_fitnesses[best_offspring] >= self.current_fitness:
				self.current_fitness = self.offspring_fitnesses[best_offspring]
				self.father = self.offsprings[best_offspring]
				self.father_was_updated = True
			# display stats
			self.logfile.write(str(self.it) + '\t' + str(self.current_fitness) + '\t' + str(self.father_was_updated) + '\t' + str(self.offspring_fitnesses) + '\n')
			if (print_every == -1 or self.it % print_every == 0):
				print(self.it, '\t', self.current_fitness, '\t', self.father_was_updated, '\t', self.offspring_fitnesses)
				self.logfile.flush()
				print('====================================================')
			if self.father_was_updated:
				#print(self.father.genome)
				self.father.save(self.folder + '/cgp_genome_' + str(self.it) + '_' + str(self.current_fitness) + '.txt')
				self.father.save(self.folder + '/best_genome_' + str(self.it) + '_' + str(self.current_fitness) + '.txt')
