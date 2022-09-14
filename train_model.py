#Importing the Libraries -----------------------------------------------------------------------------------------------
import statistics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import LSTM, Dense
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential


#Get the Dataset -------------------------------------------------------------------------------------------------------
directory =''
name ='Data'
df= pd.read_csv(directory+name+".csv", na_values = ["null"])
df.head()


# CALCULATE TIME SINCE LAST JUMP AND JUMP SIZE ----------------------------------------------------------------------------
previous_jump = []
jump_times= [] # time since last jump
skip = 1 # set to 1 to not skip vlaues
jump = 0 # this is the value of the previous large jump
diff = 0 # this is the difference between 2 adjacent currents
jump_index_time = 0
for i in range(0,len(df),skip):
    if i > 0:
        diff = abs(df['Current density'][i] - df['Current density'][i-1])
    if diff > 0.1:
        jump = diff
        jump_index_time = df['Time'][i]
    current_time = df['Time'][i]
    jump_time = current_time-jump_index_time
    previous_jump.append(jump)
    jump_times.append(jump_time)

df['Previous Jump'] = previous_jump
df['Previous Jump Time'] = jump_times
print('test stop')


#Selecting the Features ------------------------------------------------------------------------------------------------
# features = ['Current density']
features = ['Current density','Previous Jump','Previous Jump Time']

# filter noise ---------------------------------------------------------------------------------------------------------
stdev = statistics.stdev(df['Voltage'])
mean = statistics.mean(df['Voltage'])
df = df.drop(df[df.Voltage < (mean-2.5*stdev)].index)
df.reset_index(drop=True, inplace=True)

#Set Target Variable----------------------------------------------------------------------------------------------------
output_var = pd.DataFrame(df['Voltage'])


# Scaling---------------------------------------------------------------------------------------------------------------
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=df.index)


# Split into training and test sets ------------------------------------------------------------------------------------
timesplit = TimeSeriesSplit(n_splits=10, test_size=None)
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
    y_train,y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()

# Process data for LSTM ------------------------------------------------------------------------------------------------
trainX = np.array(X_train)
testX = np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build the LSTM Model -------------------------------------------------------------------------------------------------
lstm = Sequential()
lstm.add(LSTM(32,input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
epoch_number = 30
history=lstm.fit(X_train, y_train, epochs=epoch_number, batch_size=8, verbose=1, shuffle=False)
y_pred = lstm.predict(X_test)

# Save model -----------------------------------------------------------------------------------------------------------
# lstm.save("Models/" + str(name)+"_epochs"+str(epoch_number)+".h5")

# Predicted vs real output of test data Figure -------------------------------------------------------------------------
plt.figure(1)
plt.plot(y_test, label='True Value')
plt.plot(y_pred, label='LSTM Value')
plt.legend()
plt.show()

# export time vs true value & NN value ---------------------------------------------------------------------------------
export = pd.DataFrame()
export['Time'] = df['Time'][0:len(y_test)]
export['True Value'] = y_test
export['Neural Network Value'] = y_pred
pd.DataFrame(export).to_csv("export.csv",index=False)

# export losses vs epochs to csv ---------------------------------------------------------------------------------------
losses = history.history['loss']
pd.DataFrame(losses).to_csv("losses.csv",index=False)


