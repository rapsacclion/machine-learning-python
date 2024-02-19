import numpy
import math
import random

LN2 = 0.6931471805599453

training_data = {
    (0, 1, 0): [0, 0],
    (1, 0, 1): [1, 1],
    (1, 0, 0): [1, 0],
    (0, 0, 1): [0, 1]
}

for i in range(100):
    for t in range(100):
        ins = (i/100, random.random(), t/100)
        outs = [i/100, t/100]
        training_data[ins] = outs


def rec_gen(s, eq, pos=[], nv=None):
    pc = pos[:]
    if nv != None:
        pc.append(nv)
    if len(s) == 0:
        return eq(pc)
    return [rec_gen(s[1:], eq, pc, j) for j in range(s[0])]


def getSizesNDim(l, sizes=None):
    if sizes == None:
        sizes = []
    if isinstance(l, list):
        sizes.append(len(l))
        getSizesNDim(l[0], sizes)
    return sizes


def zeroes(*_):
    return 0


def ones(*_):
    return 1


class S:
    def __init__(self, s, eq=zeroes):
        self.dim = len(s)
        self.sizes = s
        self._list = rec_gen(s, eq)

    def setl(self, newlist):
        self._list = newlist
        self.sizes = getSizesNDim(newlist)
        self.dim = len(self.sizes)

    def __repr__(self):
        return self._list

    def __str__(self):
        return str(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __len__(self):
        return len(self._list)


def randominit(*_):
    return random.randint(0, 100)/100


class activFunct:
    def __init__(self, eq, inveq, derivativeeq, bounds):
        self.eq = eq
        self.inveq = inveq
        self.derivativeeq = derivativeeq
        self.bounds = bounds

    def toData(self):
        return {
            "eq": self.eq,
            "inveq": self.inveq,
            "bounds": self.bounds
        }


def sigmoid(a):
    return 1/(1+math.pow(2, -a))


def inversesigmoid(a):
    return -math.log2(1/a-1)


def sigmoidderivative(a):
    # throws a math range error in the last pow function sometimes? idk why
    return (-LN2*math.pow(2, -a))/(1+math.pow(2, 1-a)+math.pow(2, -a)**2)


def tangentderivative(a):
    return 1/(1+a*a)


sigmoidFunct = activFunct(sigmoid, inversesigmoid, sigmoidderivative, [0, 1])
tanFunct = activFunct(math.atan, math.tan,
                      tangentderivative, [-math.pi/2, math.pi/2])

class Network:
    def __init__(self, inputlayersize: int, layersarchitecture: list, outputlayersize: int, default_eq_bias=ones, default_eq_weight=zeroes, activation_funct=sigmoidFunct):
        self.arch = layersarchitecture
        self.filleq_b = default_eq_bias
        self.filleq_w = default_eq_weight
        self.activation_funct = activation_funct
        self.activations = [inputlayersize] + \
            layersarchitecture+[outputlayersize]
        biases = []
        weights = []
        i = 1
        while i < len(self.activations):
            biases.append([self.filleq_w()
                          for _ in range(self.activations[i])])
            weights.append([[self.filleq_b() for _ in range(
                self.activations[i-1])] for _ in range(self.activations[i])])
            i += 1
        self.b = biases
        self.w = weights

    def toData(self):
        return {
            "arch": self.arch,
            "filleq_b": self.filleq_b,
            "filleq_w": self.filleq_w,
            "activation_funct": self.activation_funct.toData(),
            "activations": self.activations,
            "biases": self.b,
            "weights": self.w
        }

    def evaluate(self, input_information, verbose=False):
        activation = input_information
        for idx, layer in enumerate(self.activations[:-1]):
            newactivation = []
            # THIS ZIP IS OKAY BECAUSE activation AND self.b[idx] MUST HAVE THE SAME LENGTH.
            for b, w in zip(self.b[idx], self.w[idx]):
                if verbose:  # debugging purposes
                    print(self.w)
                    print(activation)
                    print(w)
                    print([a*individual_weight for a,
                          individual_weight in zip(activation, w)])
                    print(sum([a*individual_weight for a,
                          individual_weight in zip(activation, w)]))
                    print(sum([a*individual_weight for a,
                          individual_weight in zip(activation, w)])+b)
                    print(self.activation_funct.eq(
                        sum([a*individual_weight for a, individual_weight in zip(activation, w)])+b))
                    print()
                newactivationval = sum(
                    [a*individual_weight for a, individual_weight in zip(activation, w)])
                newactivationval += b
                newactivation.append(
                    self.activation_funct.eq(newactivationval))
            activation = newactivation
        return activation


class basicTrainer:
    def __init__(self, network, trainingData):
        self.net = network
        self.data = training_data

    def getTrainingData(self, idx):
        return list(self.data)[idx]

    def getFitness(self, idx):
        ins = self.getTrainingData(idx)
        outs = self.net.evaluate(ins)
        return sum([(tdats-outss)**2 for tdats, outss in zip(self.data[ins], outs)])

    def getOverallFitness(self):
        return sum([self.getFitness(idx) for idx in range(len(self.data))])/len(self.data)

    def getDerivativeForCoordinate_WEIGHT(self, idx, layer_coordinate, lindex, lindex_in_coordinate):
        todaysTrainingData = self.getTrainingData(idx)
        activation = [todaysTrainingData]
        activation_derivatives = [[0 for _ in self.getTrainingData(idx)]]
        layer = 0
        for biases, weights in zip(self.net.b, self.net.w):
            new_activation = []
            new_activation_derivatives = []
            layerindex = 0
            for bias, weightgroup in zip(biases, weights):
                activSum = bias
                activDerivSum = 0
                i = 0
                while i < len(weightgroup):
                    isWantedCoordinate = layer == layer_coordinate and lindex == layerindex and i == lindex_in_coordinate
                    activDerivSum += weightgroup[i] * \
                        (activation_derivatives[-1][i]+isWantedCoordinate)
                    activSum += weightgroup[i]*activation[-1][i]
                    i += 1
                activ = self.net.activation_funct.eq(activSum)
                new_activation.append(activ)
                dactiv = self.net.activation_funct.derivativeeq(
                    activSum)*activDerivSum
                new_activation_derivatives.append(dactiv)
                layerindex += 1
            activation += [new_activation]
            activation_derivatives += [new_activation_derivatives]
            layer += 1
        ending_derivatives = activation_derivatives[-1]
        ending_values = activation[-1]
        return sum([2*(e_val-expected)*e_deriv for e_val, e_deriv, expected in zip(ending_values, ending_derivatives, self.data[todaysTrainingData])])

    def getDerivativeForCoordinate_BIAS(self, idx, layer_coordinate, lindex):
        todaysTrainingData = self.getTrainingData(idx)
        activation = [todaysTrainingData]
        print(activation)
        activation_derivatives = [[0 for _ in self.getTrainingData(idx)]]
        layer = 0
        # Cannot use the zip() function here; self.net.b should be offset one unit.
        for biases, weights in zip(self.net.b, self.net.w):
            new_activation = []
            new_activation_derivatives = []
            layerindex = -1
            # Probably a bad idea to use the zip() function here too.
            for bias, weightgroup in zip([0]+biases, weights):
                activSum = bias
                activDerivSum = layer == layer_coordinate and lindex == layerindex
                i = 0
                while i < len(weightgroup):
                    activDerivSum += weightgroup[i] * \
                        (activation_derivatives[-1][i])
                    activSum += weightgroup[i]*activation[-1][i]
                    i += 1
                activ = self.net.activation_funct.eq(activSum)
                new_activation.append(activ)
                dactiv = self.net.activation_funct.derivativeeq(
                    activSum)*activDerivSum
                new_activation_derivatives.append(dactiv)
                layerindex += 1
            activation += [new_activation]
            activation_derivatives += [new_activation_derivatives]
            layer += 1
        ending_derivatives = activation_derivatives[-1]
        ending_values = activation[-1]
        return sum([2*(e_val-expected)*e_deriv for e_val, e_deriv, expected in zip(ending_values, ending_derivatives, self.data[todaysTrainingData])])

    def basicTrain(self, learningRate, repetitions):
        shuffleLookup = [*range(len(self.data))]
        print(self.net.w)
        print(self.net.b)
        for i in range(repetitions):
            idx = i % len(self.data)
            random.shuffle(shuffleLookup)
            for wlayer, wlayerContents in enumerate(self.net.w):
                for wneuron, wneuronContents in enumerate(wlayerContents):
                    self.net.b[wlayer][wneuron] -= learningRate * \
                        self.getDerivativeForCoordinate_BIAS(
                            idx, wlayer, wneuron)
                    # for wconnections,_ in enumerate(wneuronContents):
                    # self.net.w[wlayer][wneuron][wconnections]-=learningRate*self.getDerivativeForCoordinate_WEIGHT(idx,wlayer,wneuron,wconnections)
            if not i % (repetitions//60+1):
                print(f"Loss overall: {self.getOverallFitness()}".ljust(50)+f"Loss on this item: {self.getFitness(
                    idx)}".ljust(50)+('#'*(60*i//repetitions)).ljust(60)+' '+str(int(100*i/repetitions))+'% done')


coolNet = Network(3, [3], 2, randominit, randominit, sigmoidFunct)
print(coolNet.evaluate([0, 0, 0]))
print(coolNet.evaluate([0, 1, 0]))  # trainingdata[0]

input()
coolTrain = basicTrainer(coolNet, training_data)

coolTrain.basicTrain(0.01, 1000)

# end of file
