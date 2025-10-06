import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


X = np.random.rand(100, 10, 1)
Y = np.random.rand(100, 1)

model = Sequential()
model.add(LSTM(50, input_shape=(10,1)))
model.add(Dropout(1))
model.compile(optimizer="Adam", loss="mse")
model.fit(X, Y, epochs=5, batch_size=16)

plt.plot(X, color="red")
plt.title("Test Graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend
plt.show





