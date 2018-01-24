import numpy as np
import copy


class netConfig:
	def __init__(self):
		self.layer = 0
		self.nodes = []
		self.lr = 0
	def setLayer(self, layerN):
		self.layer = layerN
	def setNodes(self, nodesN):
		self.nodes = copy.copy(nodesN)
	def setLR(self, LR):
		self.lr = LR

class network:
	# default output dim is 1
	# default input dim is 2
	def __init__(self, config):
		self.config = copy.deepcopy(config)
		w = []
		for i in range(self.config.layer):
			
