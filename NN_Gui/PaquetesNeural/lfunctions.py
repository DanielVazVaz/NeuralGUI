import numpy as np

## Sigmoide function
def sigmoid(x, deriv = False):

    if deriv == True:
        return sigmoid(x, deriv==False)*(1-sigmoid(x,deriv==False))
    else:
        return (1 + np.exp(-x))**-1

## ReLu function
def relu(x, deriv = False):
    if deriv == True:
        copy = x.copy()
        copy[copy<0] = 0
        copy[copy != 0] = 1
        return copy
    else:
        copy = x.copy()
        copy[copy<0] = 0
        return copy

## Simple Error Function
def error(x, deriv = False):
    real,med = x
    if deriv == True:
        return (med - real)/real.shape[0]
    if deriv == False:
        return np.mean((med - real)**2)

## Linear function
def lin(x, deriv = False):
    if deriv == True:
        copia = x.copy()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                copia[i,j] = 1
        return copia
    if deriv == False:
        return x 

## Softmax function
def soft(x, deriv = False):
    if deriv == True:
        return 1  #It is already taken into account in the cros entrp derivative
    else:
        return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)

## Función crosentropía
def crossEntrop(x, deriv = False):
    real, med = x
    if deriv ==True:
        copia = med.copy()
        for i in range(copia.shape[0]):
            for j in range(copia.shape[1]):
                if real[i,j] ==1:
                    copia[i,j] -= 1
        return copia/real.shape[0]  # Esto es el chanchullo más chanchullero de la historia de los chanchullos
    else:
        return np.sum(1/real.shape[0]*-np.log(med[real==1].reshape(med.shape[0],1))) 