from numpy import random
import numpy as np
import copy
import csv
import matplotlib.pyplot as plt
import time
import math

class Nodes:
    e = np.exp(1)
    def __init__(self, inputs):
        self.bias = random.rand()
        self.s = float()
        self.u = float()
        self.delt = float()
        self.weights = list()
        for i in range(0, inputs-1):
            self.weights.append(random.rand())
        self.oldWeights = self.weights
        self.oldBias = self.bias

    def sigmoid(s):
        #activates the weighted sum
        u = 1/(1+(Nodes.e**(-s)))
        return u
        
    def differential(u):
        #get the differential
        diff = u*(1-u)
        return diff

    def delta(self, w, deltOut, u, vohm):
        #get the delta
        diff = Nodes.differential(u)

        delt = ((w*deltOut)) *diff
        return delt

    def deltaOut(self, c, u, vohm):

        diff = Nodes.differential(u)

        delt = (c - u + vohm) * diff
        return delt
    
    def calculateS(self, row):
        #calcualte weighted sum
        self.s = self.bias

        for i in range(len(self.weights)):
            self.s += self.weights[i]*row[i]

    def updateWeights(self, LR, row, a):
        #update the weights of the node

        #list of change in weights
        changeOld = []
        for i,weight in enumerate(self.weights):
            change = weight - self.oldWeights[i]
            changeOld.append(change)
        self.oldWeights = copy.deepcopy(self.weights)
        biasChange = self.bias - self.oldBias
        self.oldBias = copy.deepcopy(self.bias)

        self.bias = self.bias + (LR*self.delt*1)
        for i in range(0, len(self.weights)):
            self.weights[i] = self.weights[i] + (LR*self.delt*row[i])
    

    def giveU(self):
        self.u = Nodes.sigmoid(self.s)

    def standard(dataSet):
        min = dataSet[0].copy()
        max = dataSet[0].copy()


        for row in dataSet:
            for i in range(0, len(min)):
                if(row[i] < min[i]):
                    min[i] = row[i]

                if(row[i] > max[i]):
                    max[i] = row[i]

        for row in dataSet:
            for i in range(0, len(row)):
                row[i] = ((0.8*((row[i] - min[i])/(max[i] - min[i]))) + 0.1)
        
        return min, max, dataSet

    def standard2(dataSet):
        min = [2.06, 1.002, 1.954, 3.694]
        max = [220.0, 80.244, 374.061, 448.1]


        for row in dataSet:
            for i in range(0, len(min)):
                if(row[i] < min[i]):
                    min[i] = row[i]

                if(row[i] > max[i]):
                    max[i] = row[i]

        for row in dataSet:
            for i in range(0, len(row)):
                row[i] = ((0.8*((row[i] - min[i])/(max[i] - min[i]))) + 0.1)
        
        return min, max, dataSet

    def standardSingle(min, max, val):
        val = ((0.8*((val - min)/(max - min))) + 0.1)
        return val
    def deStandard(min, max,val):

        value = (((val - 0.1)/0.8) * (max - min)) + min
        return value

def f(data):
    for i in data:
        print(i)

def main():
    epochs = 10
    num_nodes = 9
    inputs = 16
    LR = 0.2
    maxLR = 0.2
    nums = ((num_nodes+1) * inputs) + (num_nodes)
    #momentum
    alpha = 0.9
    
    #annealing
    q = 0.01

    #bold driver
    threshold = 4


    outNode = Nodes(num_nodes+1)
    rowForOut = []
    validRowForOut = []
    error_s = []
    epoch = []
    validationS = []
    avgDelt = np.zeros(num_nodes)
    avgBias = np.zeros(num_nodes)

    file = open("cleanedDataSet.csv")


    csvreader = csv.reader(file)

    data_set = []
    for row in csvreader:
        intRow = [float(x) for x in row]
        data_set.append(intRow)
    #shuffling the data set to make sure all the data are trained on
    random.shuffle(data_set)
    dataSet60 = round(len(data_set)*0.6)
    #standardising the data
    min, max, data_set = Nodes.standard(data_set)

    #splitting the data up into 60 20 20 (training, validation, testing)
    trainingSet = data_set[:dataSet60]
    validationSet = data_set[dataSet60:(dataSet60+round(len(data_set)*0.2))]
    testingSet = data_set[dataSet60+round(len(data_set)*0.2):dataSet60+round(len(data_set)*0.2)+round(len(data_set)*0.2)]

    nodes = list()
    for p in range(0, num_nodes):
        nodes.append(Nodes(inputs))


    lastValidation = 100000
    for i in range(0, epochs):
        errorr = 0
        validationError = 0
        avgDeltOut = 0
        avgBiasOut = 0

        
        # #bold driver
        # if((i+1) % 50 == 0):
        #     if((validationError/len(validationSet)) > (lastValidation/len(validationSet))):
        #         LR = LR*0.7
        #         print(LR)
        #     elif((((validationError/len(validationSet)) - (lastValidation/len(validationSet)))/(lastValidation/len(validationSet))) * 100 <= -threshold):
        #         LR = LR*1.05
        #         print(LR)


        for row in trainingSet:
            rowForOut = []
            for node in nodes:
                node.calculateS(row)
                node.giveU()
                rowForOut.append(node.u)
            
            outNode.calculateS(rowForOut)
            outNode.giveU()


            #weight decay
            sums = 0
            count = 0
            for node in nodes:
                for weight in node.weights:
                    sums += pow(weight,2)
                    count += 1
                sums += pow(node.bias,2)
                count += 1
            for weight in outNode.weights:
                sums += pow(weight,2)
                count += 1
            sums += pow(outNode.bias,2)
            count += 1
            ohm = 1/(2*count)*sums
            v = 1/(LR*(i+1))

            vohm = v*ohm



            outNode.delt = outNode.deltaOut(row[inputs - 1], outNode.u, vohm)

            for j in range(0, len(outNode.weights)):
                nodes[j].delt = nodes[j].delta(outNode.weights[j], outNode.delt, nodes[j].u, ohm)

            
            
            outNode.updateWeights(LR, rowForOut,alpha)

            for node in nodes:
                node.updateWeights(LR, row, alpha)

            lastValidation = validationError
        


        for row in trainingSet:
            rowForOut = []
            for node in nodes:
                node.calculateS(row)
                node.giveU()
                rowForOut.append(node.u)
            
            outNode.calculateS(rowForOut)
            outNode.giveU()

            errorr += pow((Nodes.deStandard(min[inputs - 1], max[inputs - 1], outNode.u) - Nodes.deStandard(min[inputs - 1], max[inputs - 1], row[inputs - 1])),2)


        for row in validationSet:
            validRowForOut = []
            for node in nodes:
                node.calculateS(row)
                node.giveU()
                validRowForOut.append(node.u)
            
            outNode.calculateS(validRowForOut)
            outNode.giveU()

            validationError += pow((Nodes.deStandard(min[inputs - 1], max[inputs - 1], outNode.u) - Nodes.deStandard(min[inputs - 1], max[inputs - 1], row[inputs - 1])),2)
  

        validationS.append(math.sqrt(validationError/len(validationSet)))
        error_s.append(math.sqrt(errorr/len(trainingSet)))
        epoch.append(i)
        # annealing
        complicated = 1/(1 + pow(Nodes.e, (10 - (20*(i+1)/epochs))))
        LR = q + ((maxLR - q) * (1 - complicated))
        if(i >= epochs - 1):
            print(LR)


    finalOutputs = []
    expectedOutputs = []
    for row in testingSet:
        expectedOutputs.append(Nodes.deStandard(min[inputs - 1], max[inputs - 1],row[inputs-1]))
    print("after training")
    averageError = 0
    for p,row in enumerate(testingSet):
        rowForOut.clear()
        for node in nodes:
            node.calculateS(row)
            node.giveU()
            rowForOut.append(node.u)
        
        outNode.calculateS(rowForOut)
        outNode.giveU()
        finalOutputs.append(Nodes.deStandard(min[inputs - 1], max[inputs - 1],outNode.u))
        error = abs(Nodes.deStandard(min[inputs - 1], max[inputs - 1], outNode.u) - Nodes.deStandard(min[inputs - 1], max[inputs - 1], row[inputs - 1]))

        averageError += error
        
        print("error:",error)
        print("average error:",averageError/(p+1))

    print("average:",averageError/len(testingSet),"m^3/s")
    print("min:",min)
    print("max:",max)


    print("time taken:",(time.time() - timeStart),"seconds")
    print(validationS[:5])
    print(error_s[:5])
    fig, axes = plt.subplots(2)
 
    

    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('rmse')
    axes[1].plot(epoch[1:], error_s[1:], label="training")
    axes[1].plot(epoch[1:], validationS[1:], label="validation")
    axes[1].legend(loc='best')

    axes[0].scatter(finalOutputs,expectedOutputs)
    plt.show()


timeStart = time.time()
main()




