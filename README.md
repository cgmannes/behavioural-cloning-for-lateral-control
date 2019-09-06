# Behavioural Cloning for Lateral Control of an Autonomous Vehicle

This repository contains python files for acheiving lateral control of
a vehicle in the Udacity Inc. self driving simulator. 

cnnImgPrep.py:  Performs a series of operations and manipulations to the Udacity 
dataset to achieve a normalized distribution steering angles and corresponding 
forward looking road images. The resulting histogram of the augemented data has 
a sufficient number of images in each bin such that all steering angles can be
learned.

behaviouralCloningCNN.py:  A python files that contains a two modified versions 
of a convolutional neural network (CNN) originally designed by NVIDIA that can be
trained using the data produced by cnnImgPrep.py. One CNN performs linear regression
to estimate a continuous function of steering angles based on forward looking input
road images. The second CNN performs logistic regression to estimate a discrete function
containing 100 bins of steering angles based on forward looking input images.

Although the continuous function is a better approximation of the true function to
map input road images to steering angles, the discretized nature of the binned CNN
is much faster at training. Therefore, for sufficiently small bins size, the 
performance of the categorical CNN is comparable to the continuous CNN, whilst 
requiring less training time.