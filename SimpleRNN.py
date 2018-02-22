from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Activation
from keras import optimizers
from keras.utils import plot_model
import numpy as np

from data_funcs import rnn_test_data


'''
Generate test data from a real simple RNN model, and then use these data to train a new simple
RNN network again. Observed conclusion:
(1) With preset weights, the model could predict accurate results without training. So we may
    know the best result before training.
(2) When the input data is large (> 1), the training effect is bad. The training tend to fail
    even when the initialized weights are already the best ones. Need to find out why next.
'''

# Build test data
samples = 100
u = 0.01
v = 0.01
w = 100
xt, yt = rnn_test_data(samples, u=u,  v=v, w=w)
x_train = xt.reshape(samples, 1, 1)
y_train = yt.reshape(samples, 1)
print("x_train: ", xt)
print("y_train: ", yt)

# Build simple RNN model
model = Sequential()
model.add(SimpleRNN(1, batch_input_shape=(1, 1, 1), stateful=True))
model.add(Dense(1))
model.compile(optimizer="sgd", loss="mean_squared_error")

weights = model.get_weights()
# Manually initialize weights
weights[0][0, 0] = u
weights[1][0, 0] = v
weights[3][0, 0] = w
model.set_weights(weights)
print(weights)

# Start training
# FIXME: with above input data, the more data we used to train, the bad training result we have.
# FIXME: not sure why???
history = model.fit(x=x_train[0:3], y=y_train[0:3], batch_size=1, epochs=1, shuffle=False, verbose=2)
print(model.get_weights())

print("========================================================================")
# samples-10
y_predict = model.predict(x=x_train[0:samples], batch_size=1)

# Print predicted and target value
for a, b in zip(y_predict, y_train[0:samples]):
    print('predicted: ', a, 'target: ', b)

# Print model summary
model.summary()

# Save model graph
plot_model(model, to_file='model.png', show_shapes=True)

# Print model parameters layer by layer
# for layer in model.layers:
#    weights = layer.get_weights()
#    print(weights)
