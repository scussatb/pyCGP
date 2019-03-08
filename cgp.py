import numpy as np
import random as rnd

class CGP:
	library = np.array(['add', 'sub', 'mult', 'div'])	

	class CGPNode:
		def __init__(self, c1, c2, f):
			self.c1 = c1
			self.c2 = c2
			self.function = f

	def __init__(self, genome, nInputs, nOutputs, nCols, nRows, maxArity):
		self.genome = genome
		self.nInputs = nInputs
		self.nOutputs = nOutputs
		self.nCols = nCols
		self.nRows = nRows
		self.maxGraphLength = nCols * nRows
		self.createGraph()
		self.maxArity = maxArity
	
	def createGraph(self):
		self.toEvaluate = np.zeros(self.maxGraphLength, dtype=bool)
		self.nodeOutput = np.zeros(self.maxGraphLength + self.nInputs, dtype=np.float64)
		self.nodesUsed = []
		self.outputGenes = np.zeros(self.nOutputs, dtype=np.int)
		self.nodes = []
		for i in range(0, self.nOutputs):
			self.outputGenes[i] = self.genome[len(self.genome)-self.nOutputs+i]
		i = 0
		while i < len(self.genome) - self.nOutputs:
			f = self.genome[i]
			c1 = self.genome[i + 1]
			c2 = self.genome[i + 2]
			i += 3
			self.nodes.append(self.CGPNode(c1, c2, f))
		self.nodes = np.array(self.nodes)
		self.nodeToEvaluate()
	
	def nodeToEvaluate(self):
		p = 0
		while p < self.nOutputs:
			self.toEvaluate[self.outputGenes[p] - self.nInputs] = True
			p = p + 1
		p = self.maxGraphLength - 1
		while p >= 0:
			if self.toEvaluate[p]:
				if self.nodes[p].c1 - self.nInputs >= 0:
					self.toEvaluate[self.nodes[p].c1 - self.nInputs] = True
				if self.nodes[p].c2 - self.nInputs >= 0:
					self.toEvaluate[self.nodes[p].c2 - self.nInputs] = True
				self.nodesUsed.append(p)
			p = p - 1
		self.nodesUsed = np.array(self.nodesUsed)
        
	def loadInputData(self, inputData):
		for p in range(self.nInputs): 
			self.nodeOutput[p] = inputData[p]

	def computeNode(self, x, y, f):
		if f == 0:
			return x+y
		elif f == 1:
			return x-y
		elif f == 2:
			return x*y
		elif f == 3:
			if y != 0:
				return x/y
			else:
				return x
		else: 
			return x

	def computeGraph(self):
		p = len(self.nodesUsed) - 1
		while p >= 0:
			x = self.nodes[self.nodesUsed[p]].c1
			y = self.nodes[self.nodesUsed[p]].c2
			f = self.nodes[self.nodesUsed[p]].function
			self.nodeOutput[self.nodesUsed[p] + self.nInputs] = self.computeNode(self.nodeOutput[x], self.nodeOutput[y], f)
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
		return CGP(self.genome, self.nInputs, self.nOutputs, self.nCols, self.nRows, self.maxArity)

	def mutate(self, nMutations):
		for i in range(0, nMutations):
			index = rnd.randint(0, len(self.genome) - 1)
			if index < self.nCols * self.nRows:
				# this is an internal node
				if index % (self.maxArity + 1) == 0:
					# mutate function
					self.genome[index] = rnd.randint(0, len(CGP.library) - 1)
				else:
					# mutate connection
					self.genome[index] = rnd.randint(0, self.nInputs + (index % (self.maxArity + 1) - 1) * self.nRows)
			else:
				# this is an output node
				self.genome[index] = rnd.randint(0, self.nInputs + self.nCols * self.nRows - 1)
		self.createGraph()		

	@classmethod	
	def random(cls, nInputs, nOutputs, nCols, nRows, maxArity):
		genome = np.zeros(nCols * nRows * 3 + nOutputs, dtype=int)
		gPos = 0
		for c in range(0, nCols):
			for r in range(0, nRows):
				genome[gPos] = rnd.randint(0, len(CGP.library) - 1)
				genome[gPos + 1] = rnd.randint(0, nInputs + c * nRows - 1)
				genome[gPos + 2] = rnd.randint(0, nInputs + c * nRows - 1)
				gPos = gPos + 3
		for o in range(0, nOutputs):
			genome[gPos] = rnd.randint(0, nInputs + nCols * nRows - 1)
			gPos = gPos + 1
		return CGP(genome, nInputs, nOutputs, nCols, nRows, maxArity)

	@classmethod
	def test(cls, num):
		c = CGP.random(2, 1, 2, 2, 2)
		for i in range(0, num):
			c.mutate(1)
			print(c.genome)
			print(c.run([1,2]))

