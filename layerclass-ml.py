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
    def __init__(self, eq, inveq, derivativeeq, eq_range):
        self.eq = eq
        self.inveq = inveq
        self.derivativeeq = derivativeeq
        self.eq_range = eq_range


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


class ActivationEquation:
    def __init__(self, function, inverse, derivative, derivative_inverse):
        self.funct = function
        self.invs = inverse
        self.deriv = derivative
        self.invderiv = derivative_inverse


class Layer:
    def __init__(self, size, previous_layersize, activation_eq: ActivationEquation, bias_fill=zeroes, weight_fill=zeroes):
        self.size = size
        self.psize = previous_layersize
        self.acteq = activation_eq
        self.weights = [[bias_fill(a, b) for a in range(self.psize)]
                        for b in range(self.size)]
        self.biases = [weight_fill(a) for a in range(self.size)]

    def eval(self, previous_layeractivation):
        return [
            self.acteq.funct(
                self.weights[a]
                + sum([
                    self.biases[b]*previous_layeractivation[b]
                    for b in range(self.psize)
                ])
            )
            for a in range(self.size)
        ]

    def eval_weight_derivative(self, previous_layeractivation, index):
        return [
            self.acteq.funct(
                self.weights[a]
                + sum([
                    self.biases[b]*previous_layeractivation[b]
                    for b in range(self.psize)
                ])
            )
            for a in range(self.size)
        ]

    def eval_bias_derivative(self, previous_layeractivation, index, p_layer_index):
        return [
            self.acteq.funct(
                self.weights[a]
                + sum([
                    self.biases[b]*previous_layeractivation[b]
                    for b in range(self.psize)
                ])
            )
            for a in range(self.size)
        ]
