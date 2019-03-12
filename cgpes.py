import numpy as np
from cgp import CGP
from evaluator import Evaluator

class CGPES:
	def __init__(self, num_offsprings, mutation_rate, father, evaluator):
		self.num_offsprings = num_offsprings
		self.mutation_rate = mutation_rate
		self.father = father
		self.num_mutations = int(len(self.father.genome) * self.mutation_rate)
		self.evaluator = evaluator

	def run(self, num_iteration):
		self.current_fitness = self.evaluator.evaluate(self.father, 0)
		self.father.save('genomes/cgp_genome_0_' + str(self.current_fitness) + '.txt')
		self.offsprings = np.empty(self.num_offsprings, dtype=CGP)
		self.offspring_fitnesses = np.zeros(self.num_offsprings, dtype=float)
		for self.it in range(1, num_iteration + 1):
			#generate offsprings
			for i in range(0, self.num_offsprings):
				self.offsprings[i] = self.father.clone()
				self.offsprings[i].mutate(self.num_mutations)
				self.offspring_fitnesses[i] = self.evaluator.evaluate(self.offsprings[i], self.it)
			#get the best fitness
			best_offspring = np.argmax(self.offspring_fitnesses)
			#compare to father
			self.father_was_updated = False
			if self.offspring_fitnesses[best_offspring] >= self.current_fitness:
				self.current_fitness = self.offspring_fitnesses[best_offspring]
				self.father = self.offsprings[best_offspring]
				self.father_was_updated = True
			# display stats
			print(self.it, '\t', self.current_fitness, '\t', self.father_was_updated, '\t', self.offspring_fitnesses)
			if self.father_was_updated:
				self.father.save('genomes/cgp_genome_' + str(self.it) + '_' + str(self.current_fitness) + '.txt')
