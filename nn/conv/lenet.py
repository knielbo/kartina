#!/home/knielbo/virtenvs/cv/bin/python
"""
LeNet architecture:
INPUT => CONV => TANH => POOL => CONV => TANH => POOL => FC => TANH => FC

Reference
LeCun et al. in their 1998, 
    Gradient-Based Learning Applied to Document Recognition

"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K



class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # init model
        model = 