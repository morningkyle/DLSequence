from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import numpy as np


'''
Smaller values are easier to getter better training result(eg. less epochs needed).
'''

def init_test_sequence(length):
    seq = np.array([i for i in range(length)])
    print("test sequence: ", seq)
    x = seq.reshape(length, 1, 1)
    y = seq.reshape(length, 1)
    return x, y


def create_model(n_neurons):
    # Create LSTM
    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# define LSTM configuration
length = 10
xt, yt = init_test_sequence(length)
model = create_model(3)
# Train LSTM
model.fit(xt, yt, epochs=10000, batch_size=length, verbose=2)
# Evaluate
result = model.predict(xt, batch_size=length, verbose=0)
for a, b in zip(result, yt):
    print('predicted: ', a, 'target: ', b)

print(model.summary())
