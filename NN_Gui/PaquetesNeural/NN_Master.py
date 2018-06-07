# ===================================================================== #
# -------------- General Neural Network ------------------------------- #
# ===================================================================== #
import numpy as np   #The imports must be here, apparently




# We create a class for the hidden layers:
class HiddenLayers:
    def __init__(self, firstfunction, middlefunction, lastfunction):
        self.W = []
        self.b = []
        self.f = []
        self.ff = firstfunction
        self.mf = middlefunction
        self.lf = lastfunction
    def setlayers(self, NumberOfInputs, NumberOfOutputs, NumberOfLayers, NumberOfNeurons):
        for i in range(NumberOfLayers):
            if i==0:
                self.W.append(2*np.random.random((NumberOfInputs, NumberOfNeurons)) - 1)
                self.b.append(2*np.random.random((1,NumberOfNeurons)) - 1)
                self.f.append(self.ff)
            elif i == NumberOfLayers - 1:
                self.W.append(2*np.random.random((NumberOfNeurons, NumberOfOutputs)) - 1)
                self.b.append(2*np.random.random((1,NumberOfOutputs)) - 1)
                self.f.append(self.lf)
            else:
                self.W.append(2*np.random.random((NumberOfNeurons,NumberOfNeurons)) - 1)
                self.b.append(2*np.random.random((1,NumberOfNeurons)) - 1)
                self.f.append(self.mf)











class NeuralNetwork:
    def __init__(self, Inputs, W2, b2, functions, learning_rate, Outputs = 0):
        self.Error  = 0
        self.Input  = Inputs
        self.Output = Outputs
        self.W = W2
        self.b = b2
        self.f = functions
        self.learn = learning_rate
        self.layer = []
        self.ldelta = []   # This one will be indexed backwards for the backpropagation
        self.net = []
        self.dW = []
        self.db = []
        self.ErrorTotal = 0
#        print(W)

    def Forward_Prop(self):
        self.layer = []
        self.net = []
        for i in range(len(self.W)+1):
            if i ==0:
                self.layer.append(self.Input)
                self.net.append(self.Input)
            else:
                self.net.append(np.dot(self.layer[i - 1], self.W[i-1]) + self.b[i - 1])
                self.layer.append(self.f[i-1](self.net[i]))

    def Calculate_Error(self,f):  # Esta programado para la función error más simple
        self.Error = f([self.Output, self.layer[-1]], deriv = True)
        self.ErrorTotal = f([self.Output, self.layer[-1]], deriv = False)

#        self.Error = self.layer[-1] - self.Output     # Esto es en realidad dError/ dOutput
#        self.ErrorTotal = np.mean(np.abs(self.Error)) # Este es el error correcto total. Cada uno de los errores es simplemente 1/2(Output - layer[-1])**2
# NOTA: Aquí podría usarse otro tipo de función de error. Como la Cross-entropy. Pero entonces habría que cambiar el self.Error

    def Backward_Prop(self):
        self.ldelta = []
        self.dW = []
        self.db = []
        for i in range(len(self.layer)-1): # This one does not need to go until self.layer[0], since that's the input
            if i == 0:
                self.ldelta.append(self.Error*self.f[-1](self.net[-1], deriv=True))  # It is dividing by the number of rows in Input
            else:
                Temporal_Error = np.dot(self.ldelta[i-1], self.W[-i].T)
                self.ldelta.append(Temporal_Error*self.f[-i-1](self.net[-i-1], deriv = True))

    def Actualize(self, reg = 0):
        self.dW=[]
        self.db=[]
        for i in range(len(self.W)):
            # We calculate the dW and db parameters
            self.dW.append(np.dot(self.layer[i].T, self.ldelta[-i-1]) + reg*self.W[i])
            self.db.append(np.sum(self.ldelta[-i-1], axis = 0, keepdims = True))

            # We actualize the weights
            self.W[i] -= self.learn*self.dW[-1]
            self.b[i] -= self.learn*self.db[-1]

















