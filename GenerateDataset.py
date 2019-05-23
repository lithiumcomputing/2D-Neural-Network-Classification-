##
# @file GenerateDataset.py
#
# @author jimli
# @version 23 May 2019
#
# Generates a random data set for a 2D plot.
# The data set will be used for the 2D Classification
# Neural Network program.

import numpy as np

##
# Checks if a point is in a circle.
# 
# @param x x-coordinate of the point.
# @param y y-coordinate of the point.
# @param R Radius of the circle.
#
# @return 0 if the point is outside the circle. Else, it returns 1.
def inCircle(x,y,R):
    result = x**2 + y**2 <= R**2
    if result == True:
        return 1
    else:
        return 0 # -1


##
# Checks if a point is in some arbitrary boundary.
# DEV NOTE: The boundary can be customized inside the code
# in the results variable
# 
# @param x x-coordinate of the point.
# @param y y-coordinate of the point.
# @param R Radius value (if any).
#
# @return 0 if the point is outside the boundary. Else, it returns 1.
def meetsCriteria(x,y,R=None):
    result = x**2 + y**2 <= 1.5**2 and not \
        (x**2 - y <= 1.5 and x**2 + y <= 1.5) \
        or x**2 + y**2 <= 0.5**2
    if result == True:
        return 1
    else:
        return 0 # -1

# Variables
training_set = []
labels = []

ITERATIONS = 10**5 # 1000
RADIUS = 1.0

# Generate a set of data points and their labels.
for index in range(ITERATIONS):
    point = list(np.random.uniform(-2,2,size=(2,)))
    training_set.append(point)
    label = meetsCriteria(point[0], point[1], RADIUS)
    labels.append(label)
  
# Write the entire data set into a file.
dataFile = open("dataset.txt", "w")
for index in range(ITERATIONS):
    dataFile.write("%f %f %d\n" %(training_set[index][0],
          training_set[index][1], labels[index]))
dataFile.close()