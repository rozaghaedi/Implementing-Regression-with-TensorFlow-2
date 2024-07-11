
# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Data definition
x = np.arange(-80, 81, 0.5)
y = x**2 + 5*x + 3

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Check the shapes of the training and testing sets
print(x_train.shape)
print(y_train.shape)

model_1 = keras.Sequential()
model_1.add(keras.layers.Dense(units=1024, input_shape=[1], activation='relu'))
model_1.add(keras.layers.Dense(units=1))

model_1.compile(optimizer='adam', loss='mean_squared_error')

model_1.summary()

hist_1 = model_1.fit(x_train, y_train, batch_size=64, epochs=1000, validation_data=(x_test, y_test))

# Model 2 definition
model_2 = keras.Sequential()
model_2.add(keras.layers.Dense(units=256, activation='relu', input_shape=[1]))
model_2.add(keras.layers.Dense(units=256, activation='relu'))
model_2.add(keras.layers.Dense(units=256, activation='relu'))
model_2.add(keras.layers.Dense(units=256, activation='relu'))
model_2.add(keras.layers.Dense(units=1))

model_2.compile(optimizer='adam', loss='mean_squared_error')

model_2.summary()

hist_2 = model_2.fit(x_train, y_train, batch_size=64, epochs=1000, validation_data=(x_test, y_test))

# Predictions
y_pred_1 = model_1.predict(x_test)
y_pred_2 = model_2.predict(x_test)

# Plotting
plt.scatter(x_test, y_test, c='red', linewidths=4, label='True values')
plt.scatter(x_test, y_pred_1, color='blue', label='Model 1 Predictions')
plt.scatter(x_test, y_pred_2, color='green', label='Model 2 Predictions')

plt.xlabel('x_test')
plt.ylabel('y values')
plt.legend()
plt.title('True Values vs Model Predictions')
plt.show()

