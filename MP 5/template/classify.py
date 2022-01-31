# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    weight_vals = np.zeros(len(train_set[0]))
    bias = 0
    for iter in range(max_iter):
        for image_vals, label in zip(train_set, train_labels):
            perceptron_fn = np.dot(image_vals, weight_vals) + bias

            fn_label = 0
            if (perceptron_fn > 0):
                fn_label = 1

            weight_increment = learning_rate * (label - fn_label) * image_vals
            bias_increment = learning_rate * (label - fn_label)
            weight_vals += weight_increment
            bias += bias_increment
    return weight_vals, bias

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    weight, bias = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    return_labels = []
    for image_vals in dev_set:
        perceptron_fn = np.dot(image_vals, weight) + bias

        fn_label = 0
        if (perceptron_fn > 0):
            fn_label = 1

        return_labels.append(fn_label)
    return return_labels

def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    return_labels = []
    for dev_image_vals in dev_set:
        neighbors = []
        for train_image_vals, train_label in zip(train_set, train_labels):
            dist = np.linalg.norm(np.sqrt((dev_image_vals - train_image_vals) ** 2))
            neighbors.append(tuple((dist, train_label)))
        neighbors.sort()

        k_neighbors = neighbors[:k]

        aggregate_label = 0
        for neighbor in k_neighbors:
            label_val = neighbor[1]
            if (label_val == 0):
                aggregate_label += -1
            else:
                aggregate_label += 1
        
        if (aggregate_label > 0):
            return_labels.append(1)
        else:
            return_labels.append(0)
    return return_labels
