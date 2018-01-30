import numpy as np
import matplotlib.pyplot as plt
import data, random, math, copy


def noActive(x):
    return x


def noActiveDiff(x):
    return 1.


def active(x):
    return 2. / (1. + math.e ** (- x)) - 1.


def activeInv(x):
    return - math.log(1. / (x + 1.) * 2 - 1.)


def activeDiff(x):
    actived = 2. / (1. + math.e ** (- x)) - 1
    return (1. + actived) * (1. - actived) / 2.
    # return actived * (1 - actived)


def concate3D(x, y):
    concatedData = [[0, 0, 1] for i in range(len(x))]
    for i in range(len(x)):
        concatedData[i][0] = x[i]
        concatedData[i][1] = y[i]
    return concatedData

def concate2D(x, y):
    concatedData = [[0, 0] for i in range(len(x))]
    for i in range(len(x)):
        concatedData[i][0] = x[i]
        concatedData[i][1] = y[i]
    return concatedData


class netConfig:
    def __init__(self):
        self.lr = 0     # learning rate
        self.mode = 0  # 0: Delta, 1: Perceptron, 2: BP
        self.batch = 0
        self.layer = 0
        self.inputD = 0  # input dimension
        self.nodes = []
        self.outputD = 0  # output dimension
        self.maxIter = 0
        self.momentum = 0
        self.active = None
        self.activeInv = None
        self.activeDiff = None

    def set(self, inputD=None, outputD=None, layer=None, nodes=None, lr=None, mode=None, batch=None,
            maxIter=None, activeDiff=None, active=None, activeInv=None, momentum=None):
        self.activeDiff = self.activeDiff if activeDiff == None else activeDiff
        self.activeInv = self.activeInv if activeInv == None else activeInv
        self.momentum = self.momentum if momentum == None else momentum
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
        assert (not (config.mode == 0 and config.layer > 1))
        assert (config.nodes[-1] == config.outputD)

        lastDim = config.inputD
        self.config = copy.deepcopy(config)
        self.w = []
        self.bias = []
        self.values = []
        self.inputData = []
        self.inactiveValues = []
        self.lastRoundDelta = []
        self.lastRoundBiasDelta = []
        for i in range(self.config.layer):
            self.lastRoundDelta.append(np.zeros((lastDim, self.config.nodes[i])))
            self.w.append(
                np.array([[np.random.normal(0, 0.5) for col in range(self.config.nodes[i])] for row in range(lastDim)]))
            self.values.append(np.array([0 for col in range(self.config.nodes[i])]))
            self.inactiveValues.append(np.array([0 for col in range(self.config.nodes[i])]))
            self.bias.append([np.random.normal(0, 0.5) for col in range(self.config.nodes[i])])
            self.lastRoundBiasDelta.append(np.zeros((self.config.nodes[i])))
            lastDim = self.config.nodes[i]
            # self.bias[0] = np.array([-3.1, -1.1, 3.4, -3.4, 0.89])
            # self.bias[1] = np.array([-0.5])
        self.figCount = 0

    def calError(self, x, y):
        count = 0.
        for i in range(len(x)):
            if y[i] * self.forward(x[i])[0] <= 0:
                count += 1
        return count / len(x)

    def calLoss(self, x, y):
        count = 0.
        for i in range(len(x)):
            count += 0.5 * (y[i] - self.forward(x[i])[0]) ** 2
        return count

    def forward(self, inputData):
        self.inputData = copy.deepcopy(inputData)
        # self.activeInput = np.zeros((self.config.inputD,))
        # for i in range(self.config.inputD):
        #    self.activeInput[i] = self.config.active(inputData[i])
        # print (self.activeInput)
        self.values[0] = np.dot(np.array([self.inputData]), self.w[0]).reshape((self.config.nodes[0],)) + self.bias[0]
        for i in range(self.config.nodes[0]):
            self.inactiveValues[0][i] = self.values[0][i]
            self.values[0][i] = self.config.active(self.values[0][i])
        lastDim = self.config.nodes[0]
        for k in range(1, self.config.layer):
            self.values[k] = np.dot(np.array([self.values[k - 1]]), self.w[k]).reshape((self.config.nodes[k],)) + \
                             self.bias[k]
            for i in range(self.config.nodes[k]):
                self.inactiveValues[k][i] = self.values[k][i]
                self.values[k][i] = self.config.active(self.values[k][i])
        return self.values[self.config.layer - 1]

    def backward(self, x, y):
        assert (self.config.batch <= len(y))
        trainProc = []
        if self.config.mode == 0:  # Delta rule (only for one layer and one output)
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
                    self.w[0][i][0] += delta[i] / self.config.batch
                if it % data.CHECK_INTERVAL == 0:
                    trainProc.append(self.calError(x, y))
                if it%(self.config.maxIter/50) == 0:
                    # data.scatter(np.array(x)[:, 0], np.array(x)[:, 1], y)
                    self.draw('Delta rule, iter ='+str(it))

        if self.config.mode == 1:  # Perceptron rule (only for one layer and one output)
            for it in range(self.config.maxIter):
                delta = [0 for col in range(self.config.inputD)]
                randomSamples = random.sample(range(len(x)), self.config.batch)
                for b in range(self.config.batch):
                    index = randomSamples[b]
                    if self.forward(x[index])[0] * y[b] <= 0:
                        for i in range(self.config.inputD):
                            delta[i] += y[b] * x[b][i]
                for i in range(self.config.inputD):
                    self.w[0][i][0] += delta[i] / self.config.batch
                if it % data.CHECK_INTERVAL == 0:
                    trainProc.append(self.calError(x, y))
                # if it%(self.config.maxIter/100) == 0:
                if it % 1 == 0:
                    # data.scatter(np.array(x)[:, 0], np.array(x)[:, 1], y)
                    self.draw('Delta rule, iter ='+str(it))

        if self.config.mode == 2:  # Back Propogation
            for it in range(self.config.maxIter):
                delta = []
                biasDelta = []
                lastDim = self.config.inputD
                for i in range(self.config.layer):
                    delta.append(np.zeros((lastDim, self.config.nodes[i])))
                    biasDelta.append(np.zeros((self.config.nodes[i],)))
                    lastDim = self.config.nodes[i]

                randomSamples = random.sample(range(len(x)), self.config.batch)
                for b in range(self.config.batch):
                    index = randomSamples[b]
                    self.forward(x[index])

                    # initialize delta in the last column
                    lastDelta = np.zeros((self.config.outputD,))
                    lastValue = self.values[self.config.layer - 1]
                    curInactiveValue = self.inactiveValues[self.config.layer - 1]
                    for i in range(self.config.outputD):
                        lastDelta[i] = self.config.activeDiff(curInactiveValue[i]) * (lastValue[i] - y[index])
                    # Update layer by layer
                    for i in range(self.config.layer)[::-1]:
                        # update bias on layer i
                        biasDelta[i] = biasDelta[i] - self.config.lr * lastDelta

                        # update weights on layer i
                        curDim = 0
                        curValue = None
                        curInactiveValue = None
                        if i > 0:
                            curDim = self.config.nodes[i - 1]
                            curValue = self.values[i - 1]
                            curInactiveValue = self.inactiveValues[i - 1]
                        else:
                            curDim = self.config.inputD
                            curValue = self.inputData
                            curInactiveValue = self.inputData
                        adjustW = - self.config.lr * np.dot(np.array([curValue]).T, np.array([lastDelta])).T.reshape(
                            (curDim, self.config.nodes[i]))
                        delta[i] = delta[i] + adjustW
                        curDelta = np.zeros((curDim,))
                        for j in range(curDim):
                            curDelta[j] = self.config.activeDiff(curInactiveValue[j])
                        lastDelta = curDelta * np.dot(self.w[i], np.array([lastDelta]).T).reshape((curDim,))
                if it % data.CHECK_INTERVAL == 0:
                    trainProc.append(self.calLoss(x, y))
                if (it + 1) % 2000 == 0:
                    #    #data.scatter2(x, y)
                    #    #self.draw()
                    self.config.lr = self.config.lr * 0.95
                for i in range(self.config.layer):
                    self.w[i] = self.w[i] + delta[i] / self.config.batch * (1 - self.config.momentum) + \
                                self.lastRoundDelta[i] / self.config.batch * self.config.momentum
                    self.bias[i] = self.bias[i] + biasDelta[i] / self.config.batch * (1 - self.config.momentum) + \
                                   self.lastRoundBiasDelta[i] / self.config.batch * self.config.momentum
                self.lastRoundDelta = delta
                self.lastRoundBiasDelta = biasDelta
                if it%(self.config.maxIter/100) == 0:
                    # data.scatter(np.array(x)[:, 0], np.array(x)[:, 1], y)
                    self.draw('Back Propogation rule, iter ='+str(it))
        return trainProc

    def draw(self, label=None, pause=0.0000001):
        # data.scatter(x, y, c)
        delta = 0.1
        xRange = np.arange(-10.0, 10.0, delta)
        yRange = np.arange(-10.0, 10.0, delta)
        X, Y = np.meshgrid(xRange, yRange)
        n = len(X)
        m = len(X[0])
        Z = [[0 for col in range(m)] for row in range(n)]
        for i in range(n):
            for j in range(m):
                Z[i][j] = self.forward([X[i][j], Y[i][j], 1])[0]
                '''
                if (Z[i][j] > 0):
                    plt.plot([X[i][j]], [Y[i][j]], 'ro')
                else:
                    plt.plot([X[i][j]], [Y[i][j]], 'bo')
                '''
        # print (self.w)
        for ln in lineSet:
            for coll in ln.collections:
                coll.remove()
            lineSet.clear()
        ln = plt.contour(X, Y, Z, [-0.5, 0, 0.5], colors=['r', 'black', 'b'])
        lineSet.append(ln)
        if label != None:
            plt.xlabel(label)
        # plt.show()
        plt.savefig("cache/data_mode"+str(self.config.mode)+"_%.8d.png" % self.figCount)
        self.figCount += 1
        plt.pause(pause)

if __name__ == "__main__":
    x, y, c = data.createData(1)
    Data = concate3D(x, y)
    lineSet = []

    # --- Experiment for 3.1 ---
    # config = netConfig()
    # config.set(inputD=3, outputD=1, layer=1, nodes=[1], lr=0.01, mode=0, batch=1,
    #            maxIter=100, active=noActive, activeDiff=noActiveDiff)

    # # Delta Rule
    # plt.show()
    # plt.ion()
    # net = network(config)
    # data.scatter(x, y, c)
    # net.draw('Delta rule Start')
    # trainProc1 = net.backward(Data, c)
    # data.scatter(x, y, c)
    # net.draw('Delta rule End', pause=1)
    # plt.ioff()
    # plt.close('all')
    # plt.show()
    # data.showTrainProc(1, [trainProc1], ["Delta Rule"])

    # # Perceptron Rule
    # plt.show()
    # plt.ion()
    # config.set(mode=1)
    # net1 = network(config)
    # data.scatter(x, y, c)
    # net1.draw()
    # net1.draw('Perceptron rule Start')
    # trainProc2 = net1.backward(Data, c)
    # data.scatter(x, y, c)
    # net1.draw('Perceptron rule End', pause=1)
    # plt.ioff()
    # plt.close('all')
    # data.showTrainProc(1, [trainProc2], ["Perceptron Rule"])

    # plt.show()
    # data.showTrainProc(2, [trainProc1, trainProc2], ["Delta Rule", "Perceptron Rule"])

    # --- Experiment for 3.2 ---
    config = netConfig()
    config.set(inputD=2, outputD=1, layer=2, nodes=[4, 1], lr=0.03, mode=2, batch=15, \
               maxIter=100000, momentum=0.9, active=active, activeDiff=activeDiff, activeInv=activeInv)

    # print (activeInv(config.active(10.)))

    net = network(config)

    # net.w[0] = np.array([[0.62, 0.47, 1.7, 0.92, -0.70], [1.4, -0.57, 0.042, -1.2, -0.26]])
    # net.w[1] = np.array([[-4.7], [-1.0], [4.9], [-4.3], [0.096]])


    print(net.forward([5., 5.]))
    print(net.values[0])
    print(net.values[1])

    trainProc = net.backward(Data, c)
    data.scatter(x, y, c)
    net.draw()
    print(net.w)

    data.showTrainProc(1, [trainProc], ['test'])
    plt.show()

