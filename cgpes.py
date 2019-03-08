import numpy as np
from cgp import CGP

class CGPES:
	def __init__(self, nOffsprings, mutationRate, father, evaluator):
		self.nOffsprings = nOffsprings
		self.mutationRate = mutationRate
		self.father = father
		self.nbMutations = int(len(self.father.genome) * self.mutationRate)
		self.evaluator = evaluator

	def run(self, numIt):
		self.currentFitness = self.evaluator(self.father, 0)
		self.offsprings = np.empty(self.nOffsprings, dtype=CGP)
		self.offspringFitnesses = np.zeros(self.nOffsprings, dtype=float)
		for self.it in range(1, numIt + 1):
			#generate offsprings
			for i in range(0, self.nOffsprings):
				self.offsprings[i] = self.father.clone()
				self.offsprings[i].mutate(self.nbMutations)
				self.offspringFitnesses[i] = self.evaluator(self.offsprings[i], self.it)
			#get the best fitness
			bestOff = np.argmax(self.offspringFitnesses)
			#compare to father
			self.fatherWasUpdated = False
			if self.offspringFitnesses[bestOff] >= self.currentFitness:
				self.currentFitness = self.offspringFitnesses[bestOff]
				self.father = self.offsprings[bestOff]
				self.fatherWasUpdated = True
			# display stats
			print(self.it, '\t', self.currentFitness, '\t', self.fatherWasUpdated, '\t', self.offspringFitnesses)
