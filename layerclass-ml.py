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

for i in range(1000):
    for t in range(1000):
        ins = (i/500-1, 0, t/500-1)
        outs = [i/500-1, t/500-1]
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
    try:
        return (-LN2*math.pow(2, -a))/(1+math.pow(2, 1-a)+math.pow(2, -a)**2)
    except OverflowError:
        print("Couldn't derivative")
        return 0.0000001

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
        self.weights = [[weight_fill(a, b) for a in range(self.psize)]
                        for b in range(self.size)]
        self.biases = [bias_fill(a) for a in range(self.size)]
    
    def getDescription(self):
        return f"Your regular, plain backpropogation-friendly neural network layer with dimensions {self.psize}->{self.size}"
    
    def getType(self):
        return "Generic neural network layer"

    def eval(self, previous_layeractivation):
        #print(f"Applying {self.weights} and {self.biases} to {previous_layeractivation}, got {[
        #    self.acteq.funct(
        #        self.biases[a]
        #        + sum([
        #            self.weights[a][b]*previous_layeractivation[b]
        #            for b in range(self.psize)
        #        ])
        #    )
        #    for a in range(self.size)
        #]}")
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
        outlist=[]
        for a in range(self.size):
            activa = self.acteq.deriv(
                self.biases[a]
                +sum([
                    self.weights[a][b]*previous_layeractivation[b] 
                    for b in range(self.psize)
                    ])
                )
            false_deriva = sum([
                (a == index and b == p_layer_index and layer_status[0]==layer_status[1]) 
                *previous_layeractivation[b] 
                +previous_layeractivation_derivatives[b]*self.weights[a][b]
                for b in range(self.psize)
                ])
            outlist.append(activa*false_deriva)
        return outlist

    def eval_bias_derivative(self, previous_layeractivation, previous_layeractivation_derivatives, index, layer_status):
        outlist=[]
        for a in range(self.size):
            activa=self.acteq.deriv(
                self.biases[a]
                + sum([
                    self.weights[a][b]*previous_layeractivation[b]
                    for b in range(self.psize)
                ])
            )
            false_deriva=sum([
                    self.weights[a][b]*previous_layeractivation_derivatives[b]
                    for b in range(self.psize)
                ])+ (a == index and layer_status[0]==layer_status[1])
            outlist.append(activa*false_deriva)
        return outlist

    def eval_inverse():
        pass # figure out later

sigmoidFunct = ActivationEquation(sigmoid, inversesigmoid, sigmoidderivative, zeroes, [0, 1])
tanFunct = ActivationEquation(math.atan, math.tan,
                      tangentderivative, zeroes, [-math.pi/2, math.pi/2])

class BasicNetwork:
    def __init__(self, makeup, activation_function=sigmoidFunct, bias_fills=zeroes, weight_fills=ones):
        self.layers=[]
        for item in makeup:
            self.layers.append(item[0](*(item[1:]),activation_function, bias_fill=bias_fills, weight_fill=weight_fills))
    
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
    def data_summarize(self):
        ostring=""
        for l in self.layers:
            ostring+=str(l.biases)+"  "+str(l.weights)+"  "
        return ostring

def train_network_basic(network, data_in, data_out):
    scaler=0.1
    layernum=0
    for layer in network.layers:
        layernum+=1
        bias_index=0
        for _ in layer.biases:
            derivs,activs=network.eval_net_bias_derivative(data_in,layernum,bias_index)
            #loss = sum([(a-b)**2 for a,b in zip(activs,data_out)])
            loss_deriv = sum([(a-b)*2*c for a,b,c in zip(activs,data_out,derivs)])
            #print(derivs,activs,loss,loss_deriv)
            layer.biases[bias_index]-=scaler*loss_deriv
            bias_index+=1
        weight_index=0
        for weightsection in layer.weights:
            weight_innerindex=0
            for _ in weightsection:
                derivs,activs=network.eval_net_weight_derivative(data_in,layernum,weight_index,weight_innerindex)
                #loss = sum([(a-b)**2 for a,b in zip(activs,data_out)])
                loss_deriv = sum([(a-b)*2*c for a,b,c in zip(activs,data_out,derivs)])
                #print(derivs,activs,loss,loss_deriv)
                layer.weights[weight_index][weight_innerindex]-=scaler*loss_deriv
                weight_innerindex+=1
            weight_index+=1


def get_network_loss(network,data_in,data_out):
    return sum([(a-b)**2 for a,b in zip(network.eval(data_in),data_out)])

net = BasicNetwork([(Layer,3,2),(Layer,2,2)], activation_function=tanFunct,bias_fills=zeroes, weight_fills=ones)
print(net.eval([1,1,1]))
print(net.eval_net_bias_derivative([1,1,1],0,0))
print(net.eval_net_weight_derivative([1,1,1],0,0,0))

in_options=list(training_data.keys())

print("\nStart Training\n\n")
for k in range(5000):
    datachoice=random.choice(in_options)
    datachoice_out=training_data[datachoice]
    print("Old loss:"+str(get_network_loss(net,datachoice,datachoice_out)).ljust(30)+str(net.eval([1,1,1]))+net.data_summarize()[0:25])
    train_network_basic(net,datachoice,datachoice_out)
    print("New loss:"+str(get_network_loss(net,datachoice,datachoice_out)).ljust(30)+str(net.eval([1,1,1]))+net.data_summarize()[0:25])
    #print([(l.biases,l.weights) for l in net.layers])

while True:
    try:
        print("go")
        v1=float(input("1: "))
        v2=float(input("2: "))
        v3=float(input("3: "))
        print(net.eval([v1,v2,v3]))
    except:
        print("n")