import numpy as np
import math


def rnn_test_data(length, u=1, v=1, w=1):
    num = length
    xt = np.array([i/num for i in range(num)])
    print(xt.shape)
    ht = np.zeros(100)
    zt = np.zeros(100)
    yt = np.zeros(100)
    for i in range(num):
        zt[i] = u * xt[i] + v * ht[i-1]
        ht[i] = math.tanh(zt[i])
        yt[i] = w * ht[i]
    return xt, yt


import pydot

pydot.Dot.create(pydot.Dot())

if __name__ == '__main__':
    xt, yt = rnn_test_data(100, u=1,  v=1)
    print(xt, yt)




