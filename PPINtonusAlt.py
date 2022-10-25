
import keras
import numpy as np
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf


# NORMALIZE 
def normalize(x):
    
    
    stand_x = (x - np.mean(x, axis = 0))
    stand_x /= (np.max(x, axis = 0) - np.min(x, axis = 0))
    
    maxx = np.max(x, axis = 0)
    minn = np.min(x, axis = 0)
    mean = np.mean(x, axis = 0)
    
    return stand_x, maxx, minn, mean
        

# READ AND CLEAN DATA TO MAKE TRAINABLE 
def readData():
    fileName = 'parkinsons.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')
    data = np.loadtxt(raw_data, usecols = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23), skiprows = 1, delimiter=",", dtype=np.str)
    
    # if blank, replace with "not a number"
    data[:,:][data[:,:]==""] = np.nan
    
  
    # convert data from string to float
    data = data.astype(np.float)
    
    # get the mean of a row, ignoring nan, then replace nan with mean
    # numpy function which computes the mean,ignoring NaN.
    saved_mean = np.nanmean(data[:,:])   
    data[:,:][np.isnan(data[:,:])] = saved_mean
    #create separate lists of the data and concatentate them
    #in the correct order
  
    #divide the given data into x and y
    # seperate x data
    xData = data[:,0:22]
    xData, maxx, minn, mean = normalize(xData)
    
    #add bias
    bias = np.ones((len(xData), 1))
    xData = np.concatenate((bias, xData), axis = 1)
    
    #seperate y data
    yData = data[:,22:23]
   
  
#split data into test and train
    xTrain, xTest, yTrain, yTest = train_test_split(
        xData, yData, train_size=0.75, test_size=0.25, random_state=42)
    
    
    
    return xTrain, yTrain, xTest, yTest, xData, yData 


xTrain, yTrain, xTest, yTest,xValues, yValues = readData()



trainData = np.concatenate((xTrain,yTrain),axis=1)
testData = np.concatenate((xTest,yTest),axis=1)





def readTestPoint():
    
    pointfileName = 'singleTestPoint.csv'
    print("fileName: ", pointfileName)
    raw_point = open(pointfileName, 'rt')
    
    
    
    
    
    
    
    pointData = np.loadtxt(raw_point, usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22), skiprows = 1, delimiter=",", dtype = np.str)
    
    #pointData[:,:][pointData[:,:]==" "] = np.nan
    
    pointData = pointData.astype(np.float)
    
    # convert data from string to float
    pointData = pointData.astype(np.float)
    
 
    #saved_mean = np.nanmean(pointData[:,:])   
    #pointData[:,:][np.isnan(pointData[:,:])] = saved_mean
    
    pointData, maxPoint, minPoint, meanPoint = normalize(pointData)
    
    #bias = np.ones((len(pointData),1))
    #pointData = np.concatenate((bias,pointData), axis = 1)
    
    
    return pointData

single_point_test = readTestPoint()
    






# DEVELOP NEURNAL NETWORK FRAMEWORK

model = Sequential()
model.add(Dense(23, input_shape=(196,23), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xTrain,
                    yTrain,
                    
                    epochs=100,
                    batch_size=10,
                    validation_data=(xValues, yValues))





# status: 1



"""

single_point_test = [119.992, 157.302, 74.997,0.00784,
                     .00007, .0037, 0.00554, 0.01109, 
                     .04374, .426, .02182, .0313, .02971,
                     .06545, .02211, 21.033, .414783, 
                     .815285, -4.81303, .266482, 2.301442,
                     .284654]

#single_point_test = np.array(single_point_test)


single_x_data = single_point_test[0:22]
single_x_data = normalize(single_x_data)



single_y_data = single_point_test[22:23]




data_tensor = tf.convert_to_tensor(single_x_data)


pointResults = model.predict(np.array([single_x_data]))
print ("point prediction:", pointResults)


"""

# status: 1



# COMPUTE RESULTS

"""
results
"""
results = model.evaluate(xTrain, yTrain)
print ("train:", results)
results = model.evaluate(xValues, yValues)
print ("validation:", results)
results = model.evaluate(xTest, yTest)
print ("test", results)

history_dict = history.history
print("history dict.keys():", history_dict.keys())



"""
plot loss
"""


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

fig = plt.figure()
ax = fig.add_axes([0.1,0.2,0.8,0.8])# [left, bottom, width, height]
# "bo" is for "blue dot"
ax.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
ax.plot(epochs, val_loss, 'b', label='Validation loss')
ax.set_title('Training and validation loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend(loc=(0.7, 0.2))

#ax.show()

"""
plot accuracy
"""

fig = plt.figure()
ax2 = fig.add_axes([0.12,0.2,0.8,0.8])# [left, bottom, width, height]
#plt.clf()   # clear figure
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

ax2.plot(epochs, acc, 'ro', label='Training acc')
ax2.plot(epochs, val_acc, 'b', label='Validation acc')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend(loc=(0.65, 0.1))

#ax2.show()

predClass = model.predict_classes(xTest)


model.summary()



predictResults = model.predict(xTest)

predictResults[:,0][predictResults[:,0]<0.5] = 0
predictResults[:,0][predictResults[:,0]>=0.5] = 1

truePos = 0
trueNeg = 0
falsePos = 0
falseNeg = 0

for i in range(len(predictResults)):
    if predictResults[i]==1 and yTest[i]==1:
        truePos +=1
    elif predictResults[i]==0 and yTest[i]==0:
        trueNeg +=1
    elif predictResults[i]==1 and yTest[i]==0:
        falsePos +=1
    else:
        falseNeg+=1

total = len(predClass)

accuracy = (trueNeg + truePos) / total
precision = (truePos) / (truePos + falsePos)
recall = (truePos) / (truePos + falseNeg)
error_rate = (falseNeg + falsePos) / total   

analysis = {'Accuracy': accuracy,
            'Error Rate': error_rate,
            'Precision': precision,
            'Recall' : recall} 



# PRINT RESULTS




print()

print("Do they have Parkinson's disease") 
print()
print()
print('%35s %15s' % ("Predicted No", "Predicted Yes"))
print()
print('%20s %15s %15s' % ("Actual Yes", falseNeg, truePos))
print()
print('%20s %15s %15s' % ("Actual No", trueNeg, falsePos))
print()
print()

print(analysis)

pointResults = model.predict(np.array([single_point_test]))
print ("point prediction:", pointResults)
