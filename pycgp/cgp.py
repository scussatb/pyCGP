import sys
import numpy as np
import random as rnd

class CGP: 	

	class CGPFunc:
		def __init__(self, f, name, arity, const_params=0):
			self.function = f
			self.name = name
			self.arity = arity
			self.const_params = const_params

	class CGPNode:
		def __init__(self, args, f, const_params):
			self.args = args
			self.const_params = const_params
			self.function = f

	def __init__(self, genome, num_inputs, num_outputs, num_cols, num_rows, library, recurrency_distance = 1.0, recursive = False, const_min = 0, const_max = 255, input_shape=1, dtype='float'):
		self.genome = genome.copy()
		self.recursive = recursive
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		if self.recursive:
			self.num_inputs += self.num_outputs
		self.num_cols = num_cols
		self.num_rows = num_rows
		self.max_graph_length = num_cols * num_rows
		self.library = library
		self.max_arity = 0
		self.max_const_params = 0
		self.const_min = const_min
		self.const_max = const_max
		self.recurrency_distance = recurrency_distance
		self.input_shape = input_shape
		self.dtype = dtype
		for f in self.library:
			self.max_arity = np.maximum(self.max_arity, f.arity)
			self.max_const_params = np.maximum(self.max_const_params, f.const_params)
		self.graph_created = False

	def create_graph(self):
		self.to_evaluate = np.zeros(self.max_graph_length, dtype=bool)
		if self.input_shape == 1:
			self.node_output = np.zeros(self.max_graph_length + self.num_inputs, dtype=self.dtype)
		else:
			self.node_output = np.zeros(((self.max_graph_length + self.num_inputs), ) + self.input_shape, dtype=self.dtype)
		self.nodes_used = []
		self.output_genes = np.zeros(self.num_outputs, dtype=int)
		self.nodes = np.empty(int(len(self.genome-self.num_outputs)/(1+self.max_arity+self.max_const_params)),
							  dtype=self.CGPNode)
		for i in range(0, self.num_outputs):
			self.output_genes[i] = self.genome[len(self.genome)-self.num_outputs+i]
		i = 0
		#building node list
		node_count = 0
		while i < len(self.genome) - self.num_outputs:
			f = self.genome[i]
			args = self.genome[i+1:i+1+self.max_arity]
			const_params = self.genome[i+1+self.max_arity:i+1+self.max_arity+self.max_const_params]
			i += self.max_arity + self.max_const_params + 1
			self.nodes[node_count] = self.CGPNode(args, f, const_params)
			node_count += 1
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
		for p in range(len(input_data)):
			self.node_output[p] = input_data[p]
		if False : # !!!!!! self.recursive:
			output_vals = self.read_output()
			for q in range(len(output_vals)):
				self.node_output[p+q] = output_vals[q]

	def compute_graph(self):
		#self.node_output_old = self.node_output.copy()
		p = len(self.nodes_used) - 1
		while p >= 0:
			if self.input_shape == 1:
				args = np.zeros(self.max_arity, dtype=self.dtype)
			else:
				args = np.zeros((self.max_arity, ) + self.input_shape, dtype=self.dtype)
			const_params = np.zeros(self.max_const_params, dtype=np.uint8)
			for i in range(0, self.max_arity):
				args[i] = self.node_output[self.nodes[self.nodes_used[p]].args[i]]#.copy() # removed _old here
			for i in range(0, self.max_const_params):
				const_params[i] = self.nodes[self.nodes_used[p]].const_params[i]
			f = self.library[self.nodes[self.nodes_used[p]].function].function
			self.node_output[self.nodes_used[p] + self.num_inputs] = f(args, const_params)#.copy()


			if self.input_shape == 1:
				if self.node_output[self.nodes_used[p] + self.num_inputs] != self.node_output[self.nodes_used[p] + self.num_inputs]:
					print(self.library[self.nodes[self.nodes_used[p]].function].name, ' returned NaN with ', args, ' and ', const_params)
				if (self.node_output[self.nodes_used[p] + self.num_inputs] < -1.0 or
					self.node_output[self.nodes_used[p] + self.num_inputs] > 1.0):
					print(self.library[self.nodes[self.nodes_used[p]].function].name, ' returned ', self.node_output[self.nodes_used[p] + self.num_inputs], ' with ', args)

			p = p - 1

#		for i in range(0, self.node_output.shape[0]):
#			print(str(self.node_output[i].max())+', '+str(self.node_output[i].mean()), end=', ')
#		print()

	def run(self, inputData):
		if (not self.graph_created):
			self.create_graph()

		if isinstance(inputData[0], np.ndarray) and self.input_shape != inputData[0].shape:
			self.input_shape = inputData[0].shape
			self.node_output = np.zeros(((self.max_graph_length + self.num_inputs),) + self.input_shape, dtype=self.dtype)

		self.load_input_data(inputData)
		self.compute_graph()
		return self.read_output().copy()

	def read_output(self):
		if self.input_shape == 1:
			output = np.zeros(self.num_outputs, dtype=self.dtype)
		else:
			output = np.zeros((self.num_outputs,) + self.input_shape, dtype=self.dtype)
		for p in range(0, self.num_outputs):
			output[p] = self.node_output[self.output_genes[p]].copy()

		return output

	def clone(self):
		return CGP(self.genome, self.num_inputs, self.num_outputs, self.num_cols, self.num_rows, self.library, self.recurrency_distance, self.recursive, self.const_min, self.const_max, self.input_shape, self.dtype)

	def mutate(self, num_mutationss):
		node_size = self.max_arity + self.max_const_params + 1
		for i in range(0, num_mutationss):
			index = rnd.randint(0, len(self.genome) - 1)
			if index < self.num_cols * self.num_rows * node_size:
				# this is an internal node
				if index % node_size == 0:
					# mutate function
					self.genome[index] = rnd.randint(0, len(self.library) - 1)
				elif index % node_size <= self.max_arity:
					# mutate connection
					node_index = int(index / node_size)
					col_index = int(node_index / self.num_rows)
					self.genome[index] = rnd.randint(0, self.num_inputs + col_index * self.num_rows - 1)
				else:
					# mutate const_params
					self.genome[index] = rnd.randint(self.const_min, self.const_max)


				#self.genome[index] = rnd.randint(0, self.num_inputs + (int(index / (self.max_arity + 1)) - 1) * self.num_rows)
			else:
				# this is an output node
				self.genome[index] = rnd.randint(0, self.num_inputs + self.num_cols * self.num_rows - 1)
	
	def mutate_per_gene(self, mutation_rate_nodes, mutation_rate_outputs):
		node_size = self.max_arity + self.max_const_params + 1
		for index in range(0, len(self.genome)):
			if index < self.num_cols * self.num_rows * node_size:
				# this is an internal node
				if rnd.random() < mutation_rate_nodes:
					if index % node_size == 0:
						# mutate function
						self.genome[index] = rnd.randint(0, len(self.library) - 1)
					elif index % node_size <= self.max_arity:
						# mutate connection
						node_index = int(index / node_size)
						col_index = int(node_index / self.num_rows)
						self.genome[index] = rnd.randint(0, self.num_inputs + col_index * self.num_rows - 1)
#						self.genome[index] = rnd.randint(0, int(min(self.max_graph_length + self.num_inputs - 1, (self.num_inputs + (
#									int(index /node_size) - 1) * self.num_rows) * self.recurrency_distance)))
					else:
						# mutate a const param
						self.genome[index] = rnd.randint(self.const_min, self.const_max)
						#self.genome[index] = rnd.randint(0, self.num_inputs + (int(index / (self.max_arity + 1)) - 1) * self.num_rows)
			else:

				# this is an output node
				if rnd.random() < mutation_rate_outputs:
					# this is an output node
					self.genome[index] = rnd.randint(0, self.num_inputs + self.num_cols * self.num_rows - 1)

	def goldman_mutate(self):
		has_functionnaly_mutated = False
		if not self.graph_created:
			self.create_graph()
		current_node_used = self.nodes_used.copy()
		while not has_functionnaly_mutated:
			# mutate once
			self.mutate(1)
			# build the new graph
			self.create_graph()
			# compare node used
			has_functionnaly_mutated = len(current_node_used) != len(self.nodes_used)
			i = 0
			while not has_functionnaly_mutated and i < len(current_node_used):
				has_functionnaly_mutated = current_node_used[i] != self.nodes_used[i]
				i += 1
#			if has_functionnaly_mutated:
#				print(current_node_used)
#				print(self.nodes_used)
#				print('----')
	
	def goldman_mutate_2(self):
		has_functionnaly_mutated = False
		if not self.graph_created:
			self.create_graph()
		current_node_used = self.nodes_used.copy()
		current_nodes = self.nodes.copy()
		has_functionnaly_mutated = False
		i = 0
		old_self = self.clone()
		while not has_functionnaly_mutated:
#			print(str(self)+'mutate')
			# mutate once
			self.mutate_per_gene(mutation_rate_nodes=0.15, mutation_rate_outputs=0.3)
			# build the new graph
			self.create_graph()
			input_names = [str(i) for i in range(old_self.num_inputs)]
			output_names = [str(i) for i in range(old_self.num_outputs)]
			has_functionnaly_mutated = old_self.to_function_string(input_names, output_names) != self.to_function_string(input_names, output_names)
			#print(has_functionnaly_mutated)
			# compare node used
# 			has_functionnaly_mutated = current_node_used[i] != self.nodes_used[i]
# 			if not has_functionnaly_mutated:
# 				# looking inside the node for mutations
# 				has_functionnaly_mutated = current_nodes[current_node_used[i]].function != self.nodes[self.nodes_used[i]].function
# 				for j in range(self.library[current_nodes[i].function].const_params):
# 					const_param = self.nodes[self.nodes_used[i]].const_params[j]
# 					current_const_param = current_nodes[current_node_used[i]].const_params[j]
# 					has_functionnaly_mutated = has_functionnaly_mutated or const_param != current_const_param
# 			i += 1

	def to_dot(self, file_name, input_names, output_names):
		#TODO: display const_params
		if not self.graph_created:
			self.create_graph()
		out = open(file_name, 'w')
		out.write('digraph cgp {\n')
		out.write('\tsize = "4,4";\n')
		self.dot_rec_visited_nodes = np.empty(1)
		for i in range(self.num_outputs):
			out.write('\t' + output_names[i] + ' [shape=oval];\n')
			self._write_dot_from_gene(output_names[i], self.output_genes[i], out, 0, input_names, output_names)
		out.write('}')
		out.close()

	def _write_dot_from_gene(self, to_name, pos, out, a, input_names, output_names):
		if pos < self.num_inputs:
			out.write('\t' + input_names[pos] + ' [shape=polygon,sides=6];\n')
			out.write('\t' + input_names[pos] + ' -> ' + to_name + ' [label="' + str(a) + '"];\n')
			self.dot_rec_visited_nodes = np.append(self.dot_rec_visited_nodes, [pos])
		else:
			pos -= self.num_inputs
			node_id = self.library[self.nodes[pos].function].name + '_' + str(pos)
			node_name = str(pos) + '_' + self.library[self.nodes[pos].function].name + '_'
			for i in range(self.library[self.nodes[pos].function].const_params):
				node_name += str(self.nodes[pos].const_params[i]) + '_'
			node_name = node_name[:-1]
			out.write('\t' + node_id + ' -> ' + to_name + ' [label="' + str(
					a) + '"];\n')
			if pos + self.num_inputs not in self.dot_rec_visited_nodes:
				out.write('\t' + node_id + ' [label= \"'+node_name+ '\", shape=none];\n')
				for a in range(self.library[self.nodes[pos].function].arity):
					self._write_dot_from_gene(node_id,
											  self.nodes[pos].args[a], out, a, input_names, output_names)
			self.dot_rec_visited_nodes = np.append(self.dot_rec_visited_nodes, [pos + self.num_inputs])

	def to_function_string(self, input_names, output_names):
		if not self.graph_created:
			self.create_graph()
		output = ''
		for o in range(self.num_outputs):
			output += (output_names[o] + ' = ')
			output += self._write_from_gene(self.output_genes[o], input_names, output_names)
			if o < self.num_outputs-1:
				output += ';\n'
			else:
				output += ';'
#			output += '\n'
		return output

	def _write_from_gene(self, pos, input_names, output_names):
		output = ''
		if pos < self.num_inputs:
			output += input_names[pos]
		else:
			pos -= self.num_inputs
			output += self.library[self.nodes[pos].function].name + '('
			for a in range(self.library[self.nodes[pos].function].arity):
				#print(' ', end='')
				output += self._write_from_gene(self.nodes[pos].args[a], input_names, output_names)
				if a != self.library[self.nodes[pos].function].arity - 1:
					output += ', '
				#else:
				#	print(')', end='')
			for a in range(self.library[self.nodes[pos].function].const_params):
				output += ', ' + str(self.nodes[pos].const_params[a])
			output += ')'
		return output


	@classmethod	
	def random(cls, num_inputs, num_outputs, num_cols, num_rows, library, recurrency_distance, recursive, const_min, const_max, input_shape, dtype):
		max_arity = 0
		max_const_params = 0
		if recursive:
			num_inputs += num_outputs
		for f in library:
			max_arity = np.maximum(max_arity, f.arity)
			max_const_params = np.maximum(max_const_params, f.const_params)
		node_size = 1 + max_arity + max_const_params
		genome = np.zeros(num_cols * num_rows * node_size + num_outputs, dtype=int)
		gPos = 0
		for c in range(0, num_cols):
			for r in range(0, num_rows):
				# Random function
				genome[gPos] = rnd.randint(0, len(library) - 1)
				# Random backward connections
				for a in range(max_arity):
					genome[gPos + a + 1] = rnd.randint(0, num_inputs + c * num_rows - 1)
				# Random constant parameters
				for a in range(max_const_params):
					genome[gPos + a + 1 + max_arity] = rnd.randint(const_min, const_max)
				gPos = gPos + node_size
		# Random output connections
		for o in range(0, num_outputs):
			genome[gPos] = rnd.randint(0, num_inputs + num_cols * num_rows - 1)
			gPos = gPos + 1
		if recursive:
			num_inputs -= num_outputs
		return CGP(genome, num_inputs, num_outputs, num_cols, num_rows, library, recurrency_distance, recursive, const_min, const_max, input_shape=input_shape, dtype=dtype)

	def save(self, file_name):
		out = open(file_name, 'w')
		out.write(str(self.num_inputs) + ' ')
		out.write(str(self.num_outputs) + ' ')
		out.write(str(self.num_cols) + ' ')
		out.write(str(self.num_rows) + ' ')
		# retrocomptability mechanisms:
		# recursivity is stored as last parameters
		# inputs are stored with recursive output values if recursivity is on
		# the real number of inputs will be calculated during reading
		if self.recursive:
			out.write('1\n')
		else:
			out.write('0\n')
		for g in self.genome:
			out.write(str(g) + ' ')
		out.write('\n')
		for f in self.library:
			out.write(f.name + ' ')
		out.close()

	@classmethod
	def load_from_file(cls, file_name, library, input_shape=1, dtype=int):
		inp = open(file_name, 'r')
		pams = inp.readline().split()
		genes = inp.readline().split()
		funcs = inp.readline().split()
		inp.close()
		params = np.empty(0, dtype=int)
		for p in pams:
			params = np.append(params, int(p))
		genome = np.empty(0, dtype=int)
		for g in genes:
			genome = np.append(genome, int(g))
		# recursivity management made for retrocomptability: see save for functionning
		recursive = False
		if len(params) >= 5:
			recursive = params[4] == 1
			if recursive:
				params[0] -= params[1]
		return CGP(genome, params[0], params[1], params[2], params[3], library, recursive=recursive, input_shape=input_shape, dtype=dtype)

	@classmethod
	def test(cls, num):
		c = CGP.random(2, 1, 2, 2, 2)
		for i in range(0, num):
			c.mutate(1)
			print(c.genome)
			print(c.run([1,2]))

