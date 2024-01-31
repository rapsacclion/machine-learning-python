import numpy,math,random

LN2 = 0.6931471805599453

training_data={
    (0,1,0):[0,0],
    (1,0,1):[1,1],
    (1,0,0):[1,0],
    (0,0,1):[0,1]
}

def rec_gen(s,eq,pos=[],nv=None):
  pc=pos[:]
  if nv!=None:
    pc.append(nv)
  if len(s)==0:
    return eq(pc)
  return [rec_gen(s[1:],eq,pc,j) for j in range(s[0])]

def getSizesNDim(l,sizes=None):
  if sizes==None:
    sizes=[]
  if isinstance(l,list):
    sizes.append(len(l))
    getSizesNDim(l[0],sizes)
  return sizes


def zeroes(*_):
    return 0
def ones(*_):
    return 1


class S:
  def __init__(self,s,eq=zeroes):
    self.dim=len(s)
    self.sizes=s
    self._list=rec_gen(s,eq)
  def setl(self,newlist):
    self._list=newlist
    self.sizes=getSizesNDim(newlist)
    self.dim=len(self.sizes)
  def __repr__(self):
    return self._list
  def __str__(self):
    return str(self._list)
  def __getitem__(self, i):
    return self._list[i]
  def __len__(self):
    return len(self._list)
  
def randominit(*_):
  return random.randint(0,100)/100


class activFunct:
  def __init__(self,eq,inveq,derivativeeq,bounds):
    self.eq=eq
    self.inveq=inveq
    self.derivativeeq=derivativeeq
    self.bounds=bounds
  def toData(self):
    return {
      "eq":self.eq,
      "inveq":self.inveq,
      "bounds":self.bounds
    }

def sigmoid(a):
  return 1/(1+math.pow(2,-a))

def inversesigmoid(a):
  return -math.log2(1/a-1)

def sigmoidderivative(a):
  return (-LN2*math.pow(2,-a))/(1+math.pow(2,1-a)+math.pow(2,-2*a))

def tangentderivative(a):
  return 1/(1+a*a)

sigmoidFunct=activFunct(sigmoid,inversesigmoid,sigmoidderivative,[0,1])
tanFunct=activFunct(math.tan,math.atan,tangentderivative,[-math.pi/2,math.pi/2])

#
#
# /$$$$$$$   /$$$$$$        /$$   /$$  /$$$$$$  /$$$$$$$$        /$$$$$$  /$$   /$$ /$$   /$$ /$$$$$$$$        /$$$$$$  /$$$$$$$$ /$$$$$$$$        /$$$$$$  /$$$$$$$         /$$$$$$  /$$        /$$$$$$   /$$$$$$  /$$$$$$$$
#| $$__  $$ /$$__  $$      | $$$ | $$ /$$__  $$|__  $$__/       /$$__  $$| $$  | $$| $$  | $$|__  $$__/       /$$__  $$| $$_____/| $$_____/       /$$__  $$| $$__  $$       /$$__  $$| $$       /$$__  $$ /$$__  $$| $$_____/
#| $$  \ $$| $$  \ $$      | $$$$| $$| $$  \ $$   | $$         | $$  \__/| $$  | $$| $$  | $$   | $$         | $$  \ $$| $$      | $$            | $$  \ $$| $$  \ $$      | $$  \__/| $$      | $$  \ $$| $$  \__/| $$      
#| $$  | $$| $$  | $$      | $$ $$ $$| $$  | $$   | $$         |  $$$$$$ | $$$$$$$$| $$  | $$   | $$         | $$  | $$| $$$$$   | $$$$$         | $$  | $$| $$$$$$$/      | $$      | $$      | $$  | $$|  $$$$$$ | $$$$$   
#| $$  | $$| $$  | $$      | $$  $$$$| $$  | $$   | $$          \____  $$| $$__  $$| $$  | $$   | $$         | $$  | $$| $$__/   | $$__/         | $$  | $$| $$__  $$      | $$      | $$      | $$  | $$ \____  $$| $$__/   
#| $$  | $$| $$  | $$      | $$\  $$$| $$  | $$   | $$          /$$  \ $$| $$  | $$| $$  | $$   | $$         | $$  | $$| $$      | $$            | $$  | $$| $$  \ $$      | $$    $$| $$      | $$  | $$ /$$  \ $$| $$      
#| $$$$$$$/|  $$$$$$/      | $$ \  $$|  $$$$$$/   | $$         |  $$$$$$/| $$  | $$|  $$$$$$/   | $$         |  $$$$$$/| $$      | $$            |  $$$$$$/| $$  | $$      |  $$$$$$/| $$$$$$$$|  $$$$$$/|  $$$$$$/| $$$$$$$$
#|_______/  \______/       |__/  \__/ \______/    |__/          \______/ |__/  |__/ \______/    |__/          \______/ |__/      |__/             \______/ |__/  |__/       \______/ |________/ \______/  \______/ |________/
                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#                                              I am running important AI training python program. Please do not shut off or close this computer!!! 
#                                                          And do not use it for anything either! I need the CPU to be focused on tensorflow keras algorithm training!   
#
#
#
#
#
#

class Network:
  def __init__(self,inputlayersize:int,layersarchitecture:list,outputlayersize:int,default_eq_bias=ones,default_eq_weight=zeroes,activation_funct=sigmoidFunct):
    self.arch=layersarchitecture
    self.filleq_b=default_eq_bias
    self.filleq_w=default_eq_weight
    self.activation_funct=activation_funct
    self.activations = [inputlayersize]+layersarchitecture+[outputlayersize]
    biases = []
    weights = []
    i=1
    while i<len(self.activations):
      biases.append([self.filleq_w() for _ in range(self.activations[i])])
      weights.append([[self.filleq_b() for _ in range(self.activations[i-1])] for _ in range(self.activations[i])])
      i+=1
    self.b = biases
    self.w = weights
  def toData(self):
    return {
      "arch":self.arch,
      "filleq_b":self.filleq_b,
      "filleq_w":self.filleq_w,
      "activation_funct":self.activation_funct.toData(),
      "activations":self.activations,
      "biases":self.b,
      "weights":self.w
    }
  def evaluate(self,input_information):
    activation=input_information
    for biases, weights in zip(self.b, self.w):
      new_activation=[]
      for weightgroup,bias in zip(weights,biases):
        new_activation.append(self.activation_funct.eq(bias+sum([act*weight for weight,act in zip(weightgroup,activation)])))
      activation=new_activation
    return activation


class basicTrainer:
  def __init__(self, network, trainingData):
    self.net=network
    self.data=training_data
  def getTrainingData(self,idx):
    return list(self.data)[idx]
  def getFitness(self,idx):
    ins = self.getTrainingData(idx)
    outs=self.net.evaluate(ins)
    return sum([(tdats-outss)**2 for tdats,outss in zip(self.data[ins],outs)])
  def getOverallFitness(self):
    summer=0
    for idx in range(len(self.data)):
      summer+=self.getFitness(idx)
    summer/=len(self.data)
    return summer
  def getDerivativeForCoordinate_BIAS(self,idx,layer_coordinate,lindex,lindex_in_coordinate):
    todaysTrainingData=self.getTrainingData(idx)
    activation=[todaysTrainingData]
    activation_derivatives=[[0 for _ in self.getTrainingData(idx)]]
    layer=-1
    for biases, weights in zip(self.net.b, self.net.w):
      new_activation=[]
      new_activation_derivatives=[]
      layerindex =0
      for bias, weightgroup in zip(biases,weights):
        activSum = bias
        activDerivSum = 0
        i=0
        while i<len(weightgroup):
          isWantedCoordinate=layer==layer_coordinate and lindex==layerindex and i==lindex_in_coordinate
          activDerivSum+=weightgroup[i]*(activation_derivatives[-1][i]+isWantedCoordinate)
          activSum+=weightgroup[i]*activation[-1][i]
          i+=1
        activ = self.net.activation_funct.eq(activSum)
        new_activation.append(activ)
        dactiv = self.net.activation_funct.derivativeeq(activSum)*activDerivSum
        new_activation_derivatives.append(dactiv)
        layerindex+=1
      activation+=[new_activation]
      activation_derivatives+=[new_activation_derivatives]
      layer+=1
    ending_derivatives=activation_derivatives[-1]
    return ending_derivatives
    

coolNet = Network(3,[3,5,4],2,randominit,randominit)
print(coolNet.evaluate([0,0,0]))
print(coolNet.evaluate([0,1,0])) # trainingdata[0]

coolTrain = basicTrainer(coolNet,training_data)
print(coolTrain.getFitness(0))
print(coolTrain.getOverallFitness())
print(coolTrain.getDerivativeForCoordinate_BIAS(0,0,0,0))