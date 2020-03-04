#!/home/knielbo/virtenvs/cv/bin/python
"""
ShallowNet architecture:
    INPUT => CONV => RELU => FC

"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # init model + input shape to 'channels last'
        model = Sequential()
        inputShape = (height, width, depth)

        # test for image format for input shape
        # see .keras/keras.json, comp. Theano
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        # define CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))

        # sofrmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return network architecture
        return model
