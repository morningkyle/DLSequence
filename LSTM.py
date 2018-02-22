from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import numpy as np



from numpy import array


length = 10
seq = np.array([i/float(length) for i in range(length)])
print(seq)
X = seq.reshape(length, 1, 1)
y = seq.reshape(length, 1)

# define LSTM configuration
n_neurons = 2
n_batch = length
n_epoch = 1000

# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result:
    print('%.1f' % value)