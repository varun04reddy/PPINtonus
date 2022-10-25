# -*- coding: utf-8 -*-

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
import csv



# user input

'''
name	MDVP:Fo(Hz)	MDVP:Fhi(Hz)	MDVP:Flo(Hz)	MDVP:Jitter(%)	
MDVP:Jitter(Abs)	MDVP:RAP	MDVP:PPQ	Jitter:DDP	MDVP:Shimmer	
MDVP:Shimmer(dB)	Shimmer:APQ3	Shimmer:APQ5	MDVP:APQ	Shimmer:DDA	NHR	HNR	RPDE	
DFA	spread1	spread2	D2	PPE

'''
      

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
    raw_data = open("C:/Users/varun/OneDrive/Documents/Python Scripts/PPINtonus/parkinsons.csv", 'rt')
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





# DEVELOP NEURNAL NETWORK FRAMEWORK

model = Sequential()

model.add(Dense(23, input_shape=(196,23), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xTrain,
                    yTrain,
                    
                    epochs=100,
                    batch_size=10,
                    validation_data=(xValues, yValues))




# status: 1




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

# 1 means that they have pd while 0 means that they are health





# POINT TESTING
num = 1

predArr = [0] * num
prediction = model.predict(xTest[0:num])

print(" ")
print("Model Prediction Value: ")
print(prediction)



for i in range(len(prediction)):
    if prediction[i] > 0.5:
        predArr[i] = 1
        
    else:
        predArr[i] = 0
    



finalArr = [0] * num
for x in range(len(prediction)):
    if predArr[x] == yTest[x]:
        finalArr[x] = True
        
    else:
        finalArr[x] = False
print(" ")
print("Predicted vs Actual Results: ")
print(finalArr)
        

count = 0
for i in range(len(prediction)):
    if finalArr[i] == True:
        count+=1

testedAcc = count/num

print(" ")
print("Testing Accuracy", testedAcc)


#

#single point testing




