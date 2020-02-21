#!/home/knielbo/virtenvs/cv/bin/python
"""
"""
# import the necessary packages
import numpy as np

class Perceptron:
    def __init__(self, N, alpha=.1):
        # init weight matrix and store learning rate
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha
    
    

