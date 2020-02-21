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
    
    def step(self, x):
        # apply step function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # insert column of 1's to allow trainable bias in W
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over epochs
        for epoch in np.arange(0, epochs):
            # loop over data points
            for (x, target) in zip(X, y):
                # dot product of x and W and pass through act
                p = self.step(np.dot(x, self.W))

                # only update W if p is not target
                if p != target:
                    # determine error
                    error = p - target

                    # update W
                    self.W += -self.alpha * error * x
    
    



    



