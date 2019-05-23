##
# @file 2DClassify.py
#
# @author Jim Li
# @version 23 May 2019
#
# This program demonstrates a simple Neural Network program
# that classifies points in a 2D plane.
#
# The output of this program will be a contour plot of the 
# classification boundaries and regions.
#
# Legend for Labels:
# 0 - Reject
# 1 - Accept

# Import The ML and Scientific Computing Modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os

# Disable Some Verbose Regarding CPU Instruction Compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Get the Data From the File
data = []
labels = []

dataFile = open("dataset.txt", 'r')
for line in dataFile:
    dataLine = list(map(float, line.split()))
    data.append([dataLine[0], dataLine[1]])
    labels.append(dataLine[2])
    
data = np.array(data)
labels = np.array(labels)

# Create training and validation data sets
EIGHTY_PERCENT_INDEX = int(np.round((0.8*len(data))))
train_data = data[:EIGHTY_PERCENT_INDEX]
train_labels = labels[:EIGHTY_PERCENT_INDEX]
val_data = data[EIGHTY_PERCENT_INDEX:]
val_labels = labels[EIGHTY_PERCENT_INDEX:]

# Build the Model
model = tf.keras.models.Sequential()
LAYERS = 4
NEURONS = 64
for _ in range(LAYERS):
    model.add(keras.layers.Dense(NEURONS, kernel_initializer="uniform",\
                                 activation=tf.nn.relu))
model.add(keras.layers.Dense(1,kernel_initializer="uniform",
                             activation=tf.nn.sigmoid))
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train The NN
EPOCHS = 5
BATCH_SIZE = 200
model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Plot Results
delta = 0.1
x = np.arange(-2.0, 2.0+delta, delta)
y = np.arange(-2.0, 2.0+delta, delta)
X, Y = np.meshgrid(x, y)

# Create Z array
rows, cols = X.shape
Z = np.array([[0.0 for __ in range(cols)] for _ in range(rows)])

for i in range(rows):
    for j in range(cols):
        xpt = X[i][j]
        ypt = Y[i][j]
        Z[i][j] = model.predict(np.array([[xpt,ypt]]))[0][0]

# Plot the Contour        
plt.contourf(X,Y,Z)
plt.colorbar()
plt.grid(True)
plt.show()