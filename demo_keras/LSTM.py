from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import numpy as np


def init_test_sequence(length):
    """
        Generate a test sequence. Smaller values could converge with much less epochs.
    """
    seq = np.array([i/10 for i in range(length)])
    print("test sequence: ", seq)
    x = seq.reshape(length, 1, 1)
    y = seq.reshape(length, 1)
    return x, y


def create_model(n_neurons):
    # Create LSTM
    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def main():
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


if __name__ == "__main__":
    main()
