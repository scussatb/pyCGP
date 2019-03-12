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

	def __init__(self, genome, num_inputs, num_outputs, num_cols, num_rows, library, max_arity):
		self.genome = genome
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.num_cols = num_cols
		self.num_rows = num_rows
		self.max_graph_length = num_cols * num_rows
		self.library = library
		self.max_arity = max_arity
		self.graph_created = False
	
	def create_graph(self):
		self.to_evaluate = np.zeros(self.max_graph_length, dtype=bool)
		self.node_output = np.zeros(self.max_graph_length + self.num_inputs, dtype=np.float64)
		self.nodes_used = []
		self.output_genes = np.zeros(self.num_outputs, dtype=np.int)
		self.nodes = np.empty(0, dtype=self.CGPNode)
		for i in range(0, self.num_outputs):
			self.output_genes[i] = self.genome[len(self.genome)-self.num_outputs+i]
		i = 0
		#building node list
		while i < len(self.genome) - self.num_outputs:
			f = self.genome[i]
			args = np.empty(0, dtype=int)
			for j in range(self.max_arity):
				args = np.append(args, self.genome[i+j+1])
			i += self.max_arity + 1
			self.nodes = np.append(self.nodes, self.CGPNode(args, f))
		self.node_to_evaluate()
		self.graph_created = True
	
	def node_to_evaluate(self):
		p = 0
		while p < self.num_outputs:
			if self.output_genes[p] - self.num_inputs >= 0:
				self.to_evaluate[self.output_genes[p] - self.num_inputs] = True
			p = p + 1
		p = self.max_graph_length - 1
		while p >= 0:
			if self.to_evaluate[p]:
				for i in range(0, len(self.nodes[p].args)):
					arg = self.nodes[p].args[i]
					if arg - self.num_inputs >= 0:
						self.to_evaluate[arg - self.num_inputs] = True
				self.nodes_used.append(p)
			p = p - 1
		self.nodes_used = np.array(self.nodes_used)
        
	def load_input_data(self, input_data):
		for p in range(self.num_inputs): 
			self.node_output[p] = input_data[p]

	def compute_graph(self):
		p = len(self.nodes_used) - 1
		while p >= 0:
			args = np.zeros(self.max_arity)
			for i in range(0, self.max_arity):
				args[i] = self.node_output[self.nodes[self.nodes_used[p]].args[i]] 
			f = self.library[self.nodes[self.nodes_used[p]].function].function
			self.node_output[self.nodes_used[p] + self.num_inputs] = f(args)
			p = p - 1

	def run(self, inputData):
		if not self.graph_created:
			self.create_graph()

		self.load_input_data(inputData)
		self.compute_graph()
		return self.read_output()

	def read_output(self):
		output = np.zeros(self.num_outputs)
		for p in range(0, self.num_outputs):
			output[p] = self.node_output[self.output_genes[p]]
		return output

	def clone(self):
		return CGP(self.genome, self.num_inputs, self.num_outputs, self.num_cols, self.num_rows, self.library, self.max_arity)

	def mutate(self, num_mutationss):
		for i in range(0, num_mutationss):
			index = rnd.randint(0, len(self.genome) - 1)
			if index < self.num_cols * self.num_rows * (self.max_arity + 1):
				# this is an internal node
				if index % (self.max_arity + 1) == 0:
					# mutate function
					self.genome[index] = rnd.randint(0, len(self.library) - 1)
				else:
					# mutate connection
					self.genome[index] = rnd.randint(0, self.num_inputs + (int(index / (self.max_arity + 1)) - 1) * self.num_rows)
			else:
				# this is an output node
				self.genome[index] = rnd.randint(0, self.num_inputs + self.num_cols * self.num_rows - 1)

	def to_dot(self, file_name):
		out = open(file_name, 'w')
		out.write('digraph cgp {\n')
		out.write('\tsize = "4,4";\n')
		for i in range(self.num_inputs):
			out.write('\tin' + str(i) + ' [shape=polygon,sides=5];\n')
		p = len(self.nodes_used) - 1
		while p >= 0:
			print(self.nodes_used[p])
			func = self.library[self.nodes[self.nodes_used[p]].function]
			out.write('\t' + func.name + str(self.nodes_used[p] + self.num_inputs) + ' [shape=none];')
			for i in range(func.arity):
				connect_id = self.nodes[self.nodes_used[p]].args[i]
				if connect_id < self.num_inputs:
					out.write('\tin' + str(connect_id) + ' -> ' + func.name + str(self.nodes_used[p] + self.num_inputs) + ';\n')
				else:
					connect_id -= self.num_inputs
					out.write('\t' + self.library[self.nodes[connect_id].function].name + str(connect_id + self.num_inputs) + ' -> ' + func.name + str(self.nodes_used[p] + self.num_inputs) + ';\n')
			p = p - 1
		for i in range(self.num_outputs):
			if (self.output_genes[i] < self.num_inputs):
				out.write('\tin' + str(self.output_genes[i]) + ' -> out' + str(i) + ';\n')
			else:
				out.write('\t'+ self.library[self.nodes[self.output_genes[i] - self.num_inputs].function].name + str(self.output_genes[i]) + ' -> out' + str(i) + ';\n')
		out.write('}')
		out.close()

	@classmethod	
	def random(cls, num_inputs, num_outputs, num_cols, num_rows, library, max_arity):
		genome = np.zeros(num_cols * num_rows * (max_arity+1) + num_outputs, dtype=int)
		gPos = 0
		for c in range(0, num_cols):
			for r in range(0, num_rows):
				genome[gPos] = rnd.randint(0, len(library) - 1)
				for a in range(max_arity):
					genome[gPos + a + 1] = rnd.randint(0, num_inputs + c * num_rows - 1)
				gPos = gPos + max_arity + 1
		for o in range(0, num_outputs):
			genome[gPos] = rnd.randint(0, num_inputs + num_cols * num_rows - 1)
			gPos = gPos + 1
		return CGP(genome, num_inputs, num_outputs, num_cols, num_rows, library, max_arity)

	def save(self, file_name):
		out = open(file_name, 'w')
		out.write(str(self.num_inputs) + ' ')
		out.write(str(self.num_outputs) + ' ')
		out.write(str(self.num_cols) + ' ')
		out.write(str(self.num_rows) + '\n')
		for g in self.genome:
			out.write(str(g) + ' ')
		out.close()

	@classmethod
	def load_from_file(cls, file_name, library, max_arity):
		inp = open(file_name, 'r')
		pams = inp.readline().split()
		genes = inp.readline().split()
		inp.close()
		params = np.empty(0, dtype=int)
		for p in pams:
			params = np.append(params, int(p))
		genome = np.empty(0, dtype=int)
		for g in genes:
			genome = np.append(genome, int(g))
		return CGP(genome, params[0], params[1], params[2], params[3], library, max_arity)

	@classmethod
	def test(cls, num):
		c = CGP.random(2, 1, 2, 2, 2)
		for i in range(0, num):
			c.mutate(1)
			print(c.genome)
			print(c.run([1,2]))

