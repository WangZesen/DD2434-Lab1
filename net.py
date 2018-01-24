import numpy as np
import matplotlib.pyplot as plt
import data, random, math, copy



def noActive(x):
	return x

def noActiveDiff(x):
	return 1.

def active(x):
	return 2. / (1. + math.e ** (- x)) - 1.

def activeDiff(x):
	actived = 2. / (1. + math.e ** (- x)) - 1.
	return (1. + actived) * (1. - actived) / 2.

def concate2D(x, y):
	concatedData = [[0, 0] for i in range(len(x))]
	for i in range(len(x)):
		concatedData[i][0] = x[i]
		concatedData[i][1] = y[i]
	return concatedData

class netConfig:
	def __init__(self):
		self.lr = 0
		self.mode = 0 # 0: Delta, 1: Perceptron, 2: BP
		self.batch = 0
		self.layer = 0		
		self.inputD = 0
		self.nodes = []		
		self.outputD = 0
		self.maxIter = 0 
		self.active = None
		self.activeDiff = None
	
	def set(self, inputD = None, outputD = None, layer = None, nodes = None, lr = None, mode = None, batch = None, 
		maxIter = None, activeDiff = None, active = None):
		
		self.activeDiff = self.activeDiff if activeDiff == None else activeDiff
		self.maxIter = self.maxIter if maxIter == None else maxIter
		self.outputD = self.outputD if outputD == None else outputD
		self.active = self.active if active == None else active
		self.inputD = self.inputD if inputD == None else inputD
		self.layer = self.layer if layer == None else layer
		self.nodes = self.nodes if nodes == None else nodes
		self.batch = self.batch if batch == None else batch
		self.mode = self.mode if mode == None else mode
		self.lr = self.lr if lr == None else lr

class network:
	# default output dim is 1
	# default input dim is 2
	def __init__(self, config):
		assert(not (config.mode == 0 and config.layer > 1))
		assert(config.nodes[-1] == config.outputD)
		
		lastDim = config.inputD
		self.config = copy.deepcopy(config)
		self.w = []
		self.values = []
		self.inputData = []
		for i in range(self.config.layer):
			self.w.append(np.array([[np.random.normal(0, 5) for col in range(self.config.nodes[i])] for row in range(lastDim)]))
			self.values.append(np.array([0 for col in range(self.config.nodes[i])]))
			lastDim = self.config.nodes[i]
	
	def calError(self, x, y):
		count = 0.
		for i in range(len(x)):
			if y[i] * self.forward(x[i])[0] <= 0:
				count += 1
		return count / len(x)
	
	def forward(self, inputData):
		self.inputData = copy.deepcopy(inputData)
		self.values[0] = np.dot(inputData, self.w[0])
		lastDim = self.config.nodes[0]
		for k in range(1, self.config.layer):
			self.values[k] = np.dot(self.values[k - 1], self.w[k])
			for i in range(self.config.nodes[k]):
				self.values[k][i] = self.config.active(self.values[k][i])
		return self.values[self.config.layer - 1]

	def backward(self, x, y):
		assert(self.config.batch < len(y))
		trainProc = []
		if self.config.mode == 0: # Delta rule (only for one layer and one output)
			for it in range(self.config.maxIter):
				delta = [0 for col in range(self.config.inputD)]
				randomSamples = random.sample(range(len(x)), self.config.batch)
				for b in range(self.config.batch):
					index = randomSamples[b]
					for i in range(self.config.inputD):
						count = 0
						for k in range(self.config.inputD):
							count = count + self.w[0][k][0] * x[index][k]
						count = - self.config.lr * (count - y[index]) * x[index][i]
						delta[i] += count
				for i in range(self.config.inputD):
					self.w[0][i][0] += delta[i]
				if it % data.CHECK_INTERVAL == 0:
					trainProc.append(self.calError(x, y))
		if self.config.mode == 1: # Perceptron rule (only for one layer and one output)
			for it in range(self.config.maxIter):
				delta = [0 for col in range(self.config.inputD)]
				randomSamples = random.sample(range(len(x)), self.config.batch)
				for b in range(self.config.batch):
					index = randomSamples[b]
					if self.forward(x[index])[0] * y[b] <= 0:
						for i in range(self.config.inputD):
							delta[i] += y[b] * x[b][i]
				for i in range(self.config.inputD):
					self.w[0][i][0] += delta[i]
				if it % data.CHECK_INTERVAL == 0:
					trainProc.append(self.calError(x, y))
		if self.config.mode == 2: # Back Propogation
			for it in range(self.config.maxIter):
				
				delta = []
				lastDim = self.inputD
				for i in range(self.config.layer):
					delta.append(np.zeros((lastDim, self.config.nodes[i])))
					lastDim = self.config.nodes[i]
					
				randomSamples = random.sample(range(len(x)), self.config.batch)
				for b in range(self.config.batch):
					index = randomSample[b]
					lastDelta = np.zeros((self.config.outputD,))
					outputs = self.forward(x[index])
					for i in range(self.config.outputD):
						lastDelta[i] = y[i] - outputs[i]
					for i in range(self.layer)[::-1]:
						pass
						
			
			
		return trainProc
				
	def draw(self):
		# data.scatter(x, y, c)
		delta = 0.05
		xRange = np.arange(-10.0, 10.0, delta)
		yRange = np.arange(-10.0, 10.0, delta)
		X, Y = np.meshgrid(xRange, yRange)
		n = len(X)
		m = len(X[0])
		Z = [[0 for col in range(m)] for row in range(n)]
		for i in range(n):
			for j in range(m):
				Z[i][j] = self.forward([X[i][j], Y[i][j]])[0]
		#print (self.w)
		plt.contour(X, Y, Z, [0])
		plt.show()
	

		
if __name__ == "__main__":

	x, y, c = data.createData(0)
	Data = concate2D(x, y)
	
	
	# --- Experiment for 3.1 ---
	config = netConfig()
	config.set(inputD = 2, outputD = 1, layer = 1, nodes = [1], lr = 0.01, mode = 0, batch = 1, \
	maxIter = 100, active = noActive, activeDiff = noActiveDiff)
	
	# Delta Rule	
	net = network(config)	
	data.scatter(x, y, c)
	net.draw()
	trainProc1 = net.backward(Data, c)
	data.scatter(x, y, c)
	net.draw()	
	
	# Perceptron Rule
	config.set(mode = 1)
	net1 = network(config)
	data.scatter(x, y, c)
	net1.draw()
	trainProc2 = net1.backward(Data, c)
	data.scatter(x, y, c)
	net1.draw()
	
	data.showTrainProc(2, [trainProc1, trainProc2], ["Delta Rule", "Perceptron Rule"])
	
	
	'''
	# --- Experiment for 3.2 ---
	config = netConfig()
	config.set(inputD = 2, outputD = 1, layer = 2, nodes = [3, 1], lr = 0.01, mode = 2, batch = 1, \
	maxIter = 100, active = active, activeDiff = activeDiff)
	
	net = network(config)
	data.scatter(x, y, c)
	net.draw()
	print (net.w)
	'''
