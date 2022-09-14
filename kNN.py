import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from matplotlib import pyplot as plt
import math


def eul_distance(train, test):
    distances = []
    for i in range(len(train)):
        subtract = (train-test)**2
        dist = np.sqrt(sum(subtract[i]))
        distances.append(dist)
    return distances

def topK(distances,y,k):
    # ind = np.argpartition(distances, -k)[-k:]
    ind = sorted(range(len(distances)), key=lambda sub: distances[sub])[:k] #find min k values' indices
    largest = []
    for i in range(k):
        largest.append(y[ind[i]]) #add output values of largest indices
    return largest


#Get the Dataset -------------------------------------------------------------------------------------------------------
directory =''
name ='Data'
df= pd.read_csv(directory+name+".csv", na_values = ["null"])
df.head()

#Set Target Variable ---------------------------------------------------------------------------------------------------
output_var = pd.DataFrame(df['Voltage'])

# create jump and time features ----------------------------------------------------------------------------------------
previous_jump = []
jump = []
jump_times= [] # time since last jump
skip = 1 # set to 1 to not skip vlaues
jump = 0 # this is the value of the previous large jump
diff = 0 # this is the difference between 2 adjacent currents
jump_index_time = 0
for i in range(0,len(df),skip):
    if i > 0:
        diff = abs(df['Current density'][i] - df['Current density'][i-1])
    if diff > 0.05:
        jump = diff
        jump_index_time = df['Time'][i]
    current_time = df['Time'][i]
    jump_time = current_time-jump_index_time
    previous_jump.append(jump)
    jump_times.append(jump_time)

df['Previous Jump'] = previous_jump
df['Previous Jump Time'] = jump_times


#Selecting the Features ------------------------------------------------------------------------------------------------
# features = ['Current density']
features = ['Current density','Previous Jump','Previous Jump Time']

feature_transform = pd.DataFrame(columns=features, data=df[features], index=df.index)

# Split into training and test sets  -----------------------------------------------------------------------------------
timesplit = TimeSeriesSplit(n_splits=10, test_size=None)
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
    y_train,y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()

# create numpy arrays of train and test data ---------------------------------------------------------------------------
trainX = np.array(X_train)
testX = np.array(X_test)
print('Done with data processing')


# kNN ------------------------------------------------------------------------------------------------------------------
K = 1
predictions = []
error = 0   #RSME Error
for i in range(len(testX)):
    l=eul_distance(trainX,testX[i]) # euclidean distance
    largest = topK(l,y_train,K) # outputs corresponding to minimum distances
    avg_largest = sum(largest)/len(largest)
    error = error + (avg_largest-y_test[i])**2
    predictions.append(avg_largest)
    if i % 20 == 0:
        print(i / len(testX) * 100)


error = math.sqrt(error)/len(y_test)
print("Error:")
print(error)

plt.plot(predictions)
plt.plot(y_test)
plt.show()


