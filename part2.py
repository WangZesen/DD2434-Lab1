from data import *
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


def concate2D(x, y):
	concatedData = [[0, 0] for i in range(len(x))]
	for i in range(len(x)):
		concatedData[i][0] = x[i]
		concatedData[i][1] = y[i]
	return concatedData
'''
x, y, c = createData(2)
Data = concate2D(x, y)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, ), random_state=1, verbose=True)
clf.fit(Data, c)

scatter(x, y, c);

delta = 0.05
xRange = np.arange(-15.0, 15.0, delta)
yRange = np.arange(-15.0, 15.0, delta)
X, Y = np.meshgrid(xRange, yRange)
n = len(X)
m = len(X[0])
Z = [[0 for col in range(m)] for row in range(n)]
for i in range(n):
	for j in range(m):
		Z[i][j] = clf.predict([[X[i][j], Y[i][j]]])[0]
#print (self.w)
plt.contour(X, Y, Z, [0])
plt.show()
'''

ts, X, y = timeSeries()
trainX = X[0:1000]
trainY = y[0:1000]
trainY_copy = copy.deepcopy(trainY)


testX = X[1000:1200]
testY = y[1000:1200]

expTime = 100
sigma = [0.03, 0.09, 0.18]

for k in range(3):
	trainY = copy.deepcopy(trainY_copy)
	for i in range(len(trainY)):
		trainY[i] = trainY[i] + np.random.normal(0, sigma[k])
		
	for i in range(2, 11):
		for j in range(1, 6):
			sumTest = 0.
			sumVal = 0.
			for k in range(expTime):
				clf = MLPRegressor(max_iter=10000, solver='lbfgs', alpha=10 ** (-j), hidden_layer_sizes=(i,i), random_state=1, early_stopping=True, validation_fraction = 0.1)
				clf.fit(trainX, trainY)
				#print clf.score(testX, testY)
			
				tmpX, X_val, tmpY, Y_val = train_test_split(trainX, trainY, random_state = 1, test_size = 0.1)
				sumVal = sumVal + clf.score(X_val, Y_val)
				sumTest = sumTest + clf.score(testX, testY)
			print round(sumTest / expTime, 4),
		print
	print 
'''
predictY = []
for i in range(301, 1501):
	testCase = [[ts[i - 20], ts[i - 15], ts[i - 10], ts[i - 5], ts[i]]]
	predictY.append(clf.predict(testCase))

plt.plot(range(306, 1506), ts[306:1506], label = "Ground Truth")
plt.plot(range(306, 1506), predictY, label = "Model")
plt.legend()
plt.show()
'''



