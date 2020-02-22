#!/home/knielbo/virtenvs/cv/bin/python
"""
Multilayered feedforward neural network
"""
# import the necessary packages
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=.1):
        # init list of W matrices, store architecture and learning rate
        self.W = list()
        self.layers = layers
        self.alpha = alpha
        # loop from index of first layer, stop before last two layers
        for i in np.arange(0, len(layers) - 2):
            # random init w connecting nodes in each layer
            # add extra node for trainable bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))
        
        # for the last two layer the input conenctions need a bias term, but not the output
        w = np.random.randn(layers[-2] + 1, layers[-1]) 
        self.W.append(w / np.sqrt(layers[-2]))
    
    def __repr__(self):
        # construct & return string of neural network architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
    
    def sigmoid(self, x):
        # activation function
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # derivate of the sigmoid activation function, assuming x has been passed through 
        # said function
        return x * (1 - x)