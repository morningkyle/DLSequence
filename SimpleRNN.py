from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Activation
from keras import optimizers
from keras.utils import plot_model
import numpy as np

from data_funcs import rnn_test_data


samples = 100
seq = np.array([i/float(samples) for i in range(samples)])
print(seq)
x_train = seq.reshape(samples, 1, 1)
y_train = seq.reshape(samples, 1)

xt, yt = rnn_test_data(samples, u=1,  v=1)
x_train = xt.reshape(samples, 1, 1)
y_train = yt.reshape(samples, 1)

model = Sequential()
model.add(SimpleRNN(1, batch_input_shape=(1, 1, 1), stateful=True))
model.add(Dense(1))
model.compile(optimizer="sgd", loss="mean_squared_error")

weights = model.get_weights()
# manually initialize weights
weights[0][0, 0] = 0.9
weights[1][0, 0] = 0.9
weights[3][0, 0] = 0.9
model.set_weights(weights)
print(weights)

# Start training
history = model.fit(x=x_train[0:samples-10], y=y_train[0:samples-10], batch_size=1, epochs=1, shuffle=False, verbose=2)
print(model.get_weights())

print("========================================================================")
y_predict = model.predict(x=x_train[samples-10:samples], batch_size=1)

# Print predicted and target value
for a, b in zip(y_predict, y_train[samples-10:samples]):
    print('predicted: ', a, 'target: ', b)

# Print model summary
model.summary()

# Save model graph
plot_model(model, to_file='model.png', show_shapes=True)

# Print model parameters
# for layer in model.layers:
#    weights = layer.get_weights()
#    print(weights)
