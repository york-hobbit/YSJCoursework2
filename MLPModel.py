# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 12:47:34 2023

@author: PWade
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt


#read data
data = pd.read_csv("dementia.csv")
data.head()

#data prep
data.info()

unique = data.nunique()
unique

#check for nan
check_nan = data.isnull()
check_nan
#SES MMSE have nan values


#drop Subject_ID, MRI_ID and Hand
data = data.drop(["Subject ID", "MRI ID", "Hand"], axis=1)

#change M/F to numerical values
data["M/F"] = data["M/F"].replace(["M", "F"], ["0", "1"])

#check data types
data.info()

#convert  M/F to int
data["M/F"] = data["M/F"].astype(int)

#Find data ranges
desc = data.describe()
desc

#replace nan MMSE with mode (catogorical) replace ses mean 

data["MMSE"] = data["MMSE"].fillna(data["MMSE"].mode())
data["SES"] = data["SES"].fillna(data["SES"].mean())

#replace nan MMSE with mode (catogorical) replace ses mean 

#data["MMSE"] = data["MMSE"].fillna(data["MMSE"].mode())
#drope mode did work so will drop rows for MMSE NaN
data=data.dropna(subset=["MMSE"])
data["SES"] = data["SES"].fillna(data["SES"].mean())


data.isnull().any()
#convert  M/F to int
data["M/F"] = data["M/F"].astype(int)

data.info()

#split data and lables
X = data[data.columns[1:12]]
y = data[data.columns[0]]

#normalise data x
X[:] = minmax_scale(X)

#Convert y labels to numbers and convert to hot encoding

encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values

#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=16, epochs=150)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.title("Initial run")
model.summary()

#run 2
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=16, epochs=200)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.title("2nd run")
model.summary()

#run 3
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=16, epochs=400)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.title("3rd run")
model.summary()

#run 4
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=300)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.title("4th run")
model.summary()

#run 5 add accuracy to plot
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=300)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')

plt.title("5th run")
plt.legend

model.summary()

#run 6 add extra layer
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=300)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')

plt.title("6th run")
plt.legend

model.summary()

#run 7 change optimizer
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=300)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')

plt.title("7th run")
plt.legend

model.summary()

#run 8 change optimizer
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile

model.compile(optimizer='adamax',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=300)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')

plt.title("8th run")
plt.legend

model.summary()


#run 9 change optimizer
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=300)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')

plt.title("9th run")
plt.legend

model.summary()


#run 10 keep adam optimizer increase epoc
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=500)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')

plt.title("10th run")
plt.legend

model.summary()

#run 11 keep adam optimizer and epoc change loss
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile

model.compile(optimizer='adam',
              loss='kullback_leibler_divergence',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=500)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')

plt.title("11th run")
plt.legend

model.summary()

#run 12 change optimizer back to rmsprop and keep epoc and loss
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile

model.compile(optimizer='rmsprop',
              loss='kullback_leibler_divergence',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=500)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')

plt.title("12th run")
plt.legend

model.summary()

#13th run return to initial set up with mpore neurons
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=16, epochs=150)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')
plt.title("13th run")
model.summary()

#14th repeat above inc epoch
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=16, epochs=300)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')
plt.title("14th run")
model.summary()

#15th repeat above inc epoch
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=16, epochs=600)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')
plt.title("15th run")
model.summary()

#16th repeat above inc epoch
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=16, epochs=1000)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')
plt.title("16th run")
model.summary()

#17th repeat above epoch chnage batch size 32
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=1000)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')
plt.title("17th run")
model.summary()

#18th repeat above keep epoch inc batch size
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=64, epochs=1000)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')
plt.title("18th run")
model.summary()

#19th repeat above same epoch decrease original batch size
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=8, epochs=1000)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')
plt.title("19th run")
model.summary()

#20th repeat above same epoch decrease original batch size
#split data training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
model

#compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=4, epochs=1000)

#Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#predict test
y_pred = model.predict(X_test)
y_pred

#actual and predicted
actual = np.argmax(y_test,axis=1)
predicted = np.argmax(y_pred,axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='train')
plt.title("20th run")
model.summary()