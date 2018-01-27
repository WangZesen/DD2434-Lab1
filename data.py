import matplotlib.pyplot as plt
import random, math
import numpy as np

CHECK_INTERVAL = 10
N = 100
#np.random.seed(1000)

def showTrainProc(n, results, labels):
	x = []
	for i in range(len(results[0])):
		x.append(i * CHECK_INTERVAL + 1)
		
	for i in range(n):
		plt.plot(x, results[i], label = labels[i])
	plt.legend()
	plt.show()

def scatter2(x, y):
	colors = []
	for i in range(len(y)):
		if y[i] == -1:
			colors.append('red')
		else:
			colors.append('blue')
	#print (colors)
	x = np.array(x).T.tolist()
	plt.scatter(x[0], x[1], c = colors, s = 15)

def scatter(x, y, c):
	colors = []
	for i in range(len(c)):
		if c[i] == -1:
			colors.append('red')
		else:
			colors.append('blue')
	#print (colors)
	plt.scatter(x, y, c = colors, s = 15)

def createData(kind):
	x = []
	y = []
	c = []
	if kind == 0: 
		# linear seperable
		# Setting: class 0: (-5, 5)
		#          class 1: (5, -5)
		#          variance: 3
		for i in range(N):
			x.append(np.random.normal(-5, 2))
			y.append(np.random.normal(5, 2))
			c.append(-1)
			x.append(np.random.normal(5, 2))
			y.append(np.random.normal(-5, 2))
			c.append(1)
	if kind == 1:
		# not linear seperable
		for i in range(N):
			while True:
				temX = np.random.normal(0, 5)
				temY = np.random.normal(0, 5)
				if (temX > 3 and temY > 3) or (temX < -3 and temY < -3):
					x.append(temX)
					y.append(temY)
					c.append(-1)
					break
			while True:
				temX = np.random.normal(0, 5)
				temY = np.random.normal(0, 5)
				if (temX < -3 and temY > 3) or (temX > 3 and temY < -3):
					x.append(temX)
					y.append(temY)
					c.append(1)
					break
	if kind == 2:
		# not linear seperable
		for i in range(N):
			while True:
				temX = np.random.normal(0, 5)
				temY = np.random.normal(0, 5)
				if (temX ** 2 + temY ** 2 < 10):
					x.append(temX)
					y.append(temY)
					c.append(-1)
					break
			while True:
				temX = np.random.normal(0, 5)
				temY = np.random.normal(0, 5)
				if (temX ** 2 + temY ** 2 > 30):
					x.append(temX)
					y.append(temY)
					c.append(1)
					break					
	# shuffle
	for i in range(20000):
		id1 = random.randint(0, 2 * N - 1)
		id2 = random.randint(0, 2 * N - 1)
		x[id1], x[id2] = x[id2], x[id1]
		y[id1], y[id2] = y[id2], y[id1]
		c[id1], c[id2] = c[id2], c[id1]

	return x, y, c

# For debug
if __name__ == "__main__":
	x, y, c = createData(0)
	#scatter(x, y, c)
	showTrainProc(2, [[1,2,3,4],[5,6,7,8]], ["line A", "line B"])

