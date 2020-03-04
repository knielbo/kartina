#!/home/knielbo/virtenvs/cv/bin/python
"""
ShallowNet architecture:
    INPUT => CONC => RELU => FC

"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # init model + input shape to 'channels last'