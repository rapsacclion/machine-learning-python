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


def sigmoid(a):
    return 1/(1+math.pow(2, -a))


def inversesigmoid(a):
    return -math.log2(1/a-1)


def sigmoidderivative(a):
    # throws a math range error in the last pow function sometimes? idk why
    return (-LN2*math.pow(2, -a))/(1+math.pow(2, 1-a)+math.pow(2, -a)**2)


def tangentderivative(a):
    return 1/(1+a*a)




class ActivationEquation:
    def __init__(self, function, inverse, derivative, derivative_inverse, bounds=[0,0]):
        self.funct = function
        self.invs = inverse
        self.deriv = derivative
        self.invderiv = derivative_inverse
        self.bounds=bounds


class Layer:
    def __init__(self, previous_layersize, size, activation_eq: ActivationEquation, bias_fill=zeroes, weight_fill=ones):
        self.size = size
        self.psize = previous_layersize
        self.acteq = activation_eq
        self.weights = [[bias_fill(a, b) for a in range(self.psize)]
                        for b in range(self.size)]
        self.biases = [weight_fill(a) for a in range(self.size)]
    
    def getDescription(self):
        return f"Your regular, plain backpropogation-friendly neural network layer with dimensions {self.psize}->{self.size}"
    
    def getType(self):
        return "Generic neural network layer"

    def eval(self, previous_layeractivation):
        return [
            self.acteq.funct(
                self.biases[a]
                + sum([
                    self.weights[a][b]*previous_layeractivation[b]
                    for b in range(self.psize)
                ])
            )
            for a in range(self.size)
        ]

    def eval_weight_derivative(self, previous_layeractivation, previous_layeractivation_derivatives, index, p_layer_index, layer_status):
        return [
            self.acteq.funct(
                self.biases[a]
                + sum([
                    self.weights[a][b]*previous_layeractivation[b]
                    for b in range(self.psize)
                ])
            )
            * (
                sum([
                    (a == index and b == p_layer_index and layer_status[0]==layer_status[1]) *
                    previous_layeractivation[b]
                    + self.weights[a][b]*previous_layeractivation_derivatives[b]
                    for b in range(self.psize)
                ])
                + 0
            )
            for a in range(self.size)
        ]

    def eval_bias_derivative(self, previous_layeractivation, previous_layeractivation_derivatives, index, layer_status):
        return [
            self.acteq.funct(
                self.biases[a]
                + sum([
                    self.weights[a][b]*previous_layeractivation[b]
                    for b in range(self.psize)
                ])
            )
            * (
                sum([
                    self.weights[a][b]*previous_layeractivation_derivatives[b]
                    for b in range(self.psize)
                ])
                + (a == index and layer_status[0]==layer_status[1])
            )
            for a in range(self.size)
        ]

    def eval_inverse():
        pass # figure out later


class BasicNetwork:
    def __init__(self, makeup, activation_function=sigmoidFunct):
        self.layers=[]
        for item in makeup:
            self.layers.append(item[0](*(item[1:]),activation_function))
    
    def getType(self):
        return "Basic layered neural network"

    def printContents(self):
        message=self.getType()+'\n'
        for i in self.layers:
            message+="  "+i.getType()+'\n'
            message+="    "+i.getDescription()+'\n'
        print(message)

    def eval(self,input_values):
        current_activation=input_values
        for layer in self.layers:
            current_activation = layer.eval(current_activation)
        return current_activation
    def eval_net_weight_derivative(self,input_values,layerindex,index,playerindex):
        current_activation=input_values
        current_activation_derivatives=[0 for _ in current_activation]

        clayerindex=0
        for layer in self.layers:
            current_activation_derivatives=layer.eval_weight_derivative(current_activation,current_activation_derivatives,index,playerindex,[clayerindex,layerindex])
            current_activation = layer.eval(current_activation)
            clayerindex+=1
        return (current_activation_derivatives,current_activation)
    def eval_net_bias_derivative(self,input_values,layerindex,index):
        current_activation=input_values
        current_activation_derivatives=[0 for _ in current_activation]

        clayerindex=0
        for layer in self.layers:
            current_activation_derivatives=layer.eval_bias_derivative(current_activation,current_activation_derivatives,index,[clayerindex,layerindex])
            current_activation = layer.eval(current_activation)
            clayerindex+=1
        return (current_activation_derivatives,current_activation)


sigmoidFunct = ActivationEquation(sigmoid, inversesigmoid, sigmoidderivative, zeroes, [0, 1])
tanFunct = ActivationEquation(math.atan, math.tan,
                      tangentderivative, zeroes, [-math.pi/2, math.pi/2])

net = BasicNetwork([(Layer,3,4),(Layer,4,3),(Layer,3,2)])