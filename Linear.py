from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Activation
from keras import optimizers
import numpy as np

x_train = np.array([0, 2, 4, 6, 8, 10])
y_train = np.array([2, 4, 6, 8, 10, 12])

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.compile(optimizer='adam', loss='mse')
model.fit(x=x_train, y=y_train, epochs=3000, batch_size=1)

x_test = np.array([100, 101, 102])
y = model.predict(x_test)
print(y)


