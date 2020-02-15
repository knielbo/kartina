#!/home/knielbo/virtenvs/cv/bin/python
"""
"""
import os
import numpy as np
import cv2

class SimpleDatasetLoader():
    def __init__(self, preprocessors=None):
        # store the image preprocessors
        self.preprocessors = preprocessors

        # if preprocessors are None, initialize empty list
        if self.preprocessors is None:
            self.preprocessors = list()
        
    def load(self, imagePaths, verbose=-1):
        # initialize list of features and labels
        data = list()
        labels = list()
        for (i, imagePath) in enumerate(imagePaths):
            # load image and extract class label
            # assumes /path/2/image/{class}/{image}.jpg path
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # check preprocessors value
            if self.preprocessors is not None:
                # loop over preprocessors and apply
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            data.append(image)
            labels.append(label)

            # show update for every 'verbose' images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] preprocessed {}/{}".format(i+1, len(imagePaths)))