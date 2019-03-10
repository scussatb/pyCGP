import numpy as np
import random as rnd

class CGP: 	

	class CGPFunc:
		def __init__(self, f, name, arity):
			self.function = f
			self.name = name
			self.arity = arity 

	class CGPNode:
		def __init__(self, args, f):
			self.args = args
			self.function = f

	def __init__(self, genome, nInputs, nOutputs, nCols, nRows, library, maxArity):
		self.genome = genome
		self.nInputs = nInputs
		self.nOutputs = nOutputs
		self.nCols = nCols
		self.nRows = nRows
		self.maxGraphLength = nCols * nRows
		self.library = library
		self.maxArity = maxArity
		self.createGraph()
	
	def createGraph(self):
		self.toEvaluate = np.zeros(self.maxGraphLength, dtype=bool)
		self.nodeOutput = np.zeros(self.maxGraphLength + self.nInputs, dtype=np.float64)
		self.nodesUsed = []
		self.outputGenes = np.zeros(self.nOutputs, dtype=np.int)
		self.nodes = np.empty(0, dtype=self.CGPNode)
		for i in range(0, self.nOutputs):
			self.outputGenes[i] = self.genome[len(self.genome)-self.nOutputs+i]
		i = 0
		while i < len(self.genome) - self.nOutputs:
			f = self.genome[i]
			args = np.empty(0, dtype=int)
			for j in range(0, self.maxArity):
				args = np.append(args, self.genome[i+j+1])
			i += self.maxArity + 1
			self.nodes = np.append(self.nodes, self.CGPNode(args, f))
		self.nodeToEvaluate()
	
	def nodeToEvaluate(self):
		p = 0
		while p < self.nOutputs:
			self.toEvaluate[self.outputGenes[p] - self.nInputs] = True
			p = p + 1
		p = self.maxGraphLength - 1
		while p >= 0:
			if self.toEvaluate[p]:
				for i in range(0, len(self.nodes[p].args)):
					arg = self.nodes[p].args[i]
					if arg - self.nInputs >= 0:
						self.toEvaluate[arg - self.nInputs] = True
				self.nodesUsed.append(p)
			p = p - 1
		self.nodesUsed = np.array(self.nodesUsed)
        
	def loadInputData(self, inputData):
		for p in range(self.nInputs): 
			self.nodeOutput[p] = inputData[p]

	def computeGraph(self):
		p = len(self.nodesUsed) - 1
		while p >= 0:
			args = np.zeros(self.maxArity)
			for i in range(0, self.maxArity):
				args[i] = self.nodeOutput[self.nodes[self.nodesUsed[p]].args[i]] 
			f = self.library[self.nodes[self.nodesUsed[p]].function].function
			self.nodeOutput[self.nodesUsed[p] + self.nInputs] = f(args)
			p = p - 1

	def run(self, inputData):
		self.loadInputData(inputData)
		self.computeGraph()
		return self.readOutput()

	def readOutput(self):
		output = np.zeros(self.nOutputs)
		for p in range(0, self.nOutputs):
			output[p] = self.nodeOutput[self.outputGenes[p]]
		return output

	def clone(self):
		return CGP(self.genome, self.nInputs, self.nOutputs, self.nCols, self.nRows, self.library, self.maxArity)

	def mutate(self, nMutations):
		for i in range(0, nMutations):
			index = rnd.randint(0, len(self.genome) - 1)
			if index < self.nCols * self.nRows * (self.maxArity + 1):
				# this is an internal node
				if index % (self.maxArity + 1) == 0:
					# mutate function
					self.genome[index] = rnd.randint(0, len(self.library) - 1)
				else:
					# mutate connection
					self.genome[index] = rnd.randint(0, self.nInputs + (index % (self.maxArity + 1) - 1) * self.nRows)
			else:
				# this is an output node
				self.genome[index] = rnd.randint(0, self.nInputs + self.nCols * self.nRows - 1)
		self.createGraph()		

	@classmethod	
	def random(cls, nInputs, nOutputs, nCols, nRows, library, maxArity):
		genome = np.zeros(nCols * nRows * (maxArity+1) + nOutputs, dtype=int)
		gPos = 0
		for c in range(0, nCols):
			for r in range(0, nRows):
				genome[gPos] = rnd.randint(0, len(library) - 1)
				genome[gPos + 1] = rnd.randint(0, nInputs + c * nRows - 1)
				genome[gPos + 2] = rnd.randint(0, nInputs + c * nRows - 1)
				gPos = gPos + 3
		for o in range(0, nOutputs):
			genome[gPos] = rnd.randint(0, nInputs + nCols * nRows - 1)
			gPos = gPos + 1
		return CGP(genome, nInputs, nOutputs, nCols, nRows, library, maxArity)

	@classmethod
	def test(cls, num):
		c = CGP.random(2, 1, 2, 2, 2)
		for i in range(0, num):
			c.mutate(1)
			print(c.genome)
			print(c.run([1,2]))

