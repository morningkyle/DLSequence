"""
Generate test data from a real simple RNN model, and then use these data to train a new simple RNN
network again. Observed conclusion:
(1) With preset weights, the model could predict accurate results without training. So we may know
    the best parameters before training.
(2) When the input data is large (> 1), the training effect is bad. This is because the output of
    each layer is very easy to saturate, and the accuracy of float32 is lost.
"""
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.utils import plot_model
from keras.callbacks import CSVLogger

from data_funcs import rnn_test_data


# Build test data
u = 0.01
v = 0.01
w = 10
samples = 100
xt, yt = rnn_test_data(samples, u=u,  v=v, w=w)
x_train = xt.reshape(samples, 1, 1)
y_train = yt.reshape(samples, 1)
print("x_train: ", xt)
print("y_train: ", yt)

# Build simple a RNN model
model = Sequential()
model.add(SimpleRNN(1, batch_input_shape=(1, 1, 1), stateful=True))
model.add(Dense(1))
# ADAM optimizer has much better performance than SGD in this model
model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])

weights = model.get_weights()
# Demo how to manually initialize weights for this model
weights[0][0, 0] = u
weights[1][0, 0] = v
weights[3][0, 0] = w
model.set_weights(weights)
print(weights)

# Start training
checkpoint = CSVLogger('log.csv', append=True, separator=';')
# checkpoint = ModelCheckpoint(filepath='weights.{epoch:02d}.hdf5', verbose=1)
# checkpoint = MyCallback()
for i in range(3):
    history = model.fit(x=x_train[0:samples-10], y=y_train[0:samples-10], batch_size=1, epochs=1, verbose=2,
                        callbacks=[checkpoint], shuffle=False)
    model.reset_states()
print(model.get_weights())

print("========================================================================")
# Let's check our prediction performance
y_predict = model.predict(x=x_train[samples-10:samples], batch_size=1)

# Print predicted and target value
for a, b in zip(y_predict, y_train[samples-10:samples]):
    print('predicted: ', a, 'target: ', b)

# Print model summary
model.summary()

# Save model graph
plot_model(model, to_file='model.png', show_shapes=True)

# Print model parameters layer by layer
# for layer in model.layers:
#    weights = layer.get_weights()
#    print(weights)
