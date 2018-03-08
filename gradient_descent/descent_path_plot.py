import numpy as np
# enabling 3D plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

from gradient_descent.steepest_gradient_descent import steepest_gradient_descent, f


xk = 3
yk = 3
xhist = [xk]
yhist = [yk]
iter = 0
should_stop = False
while iter < 1000 and should_stop is False:
    xk, yk, should_stop = steepest_gradient_descent(xk, yk)
    xhist.append(xk)
    yhist.append(yk)
    iter += 1
    print("iteration: {0}, x:{1}, y:{1}".format(iter, xk, yk))

x = np.arange(-3.0, 3.0, 0.01)
y = np.arange(-3.0, 3.0, 0.01)
X, Y = np.meshgrid(x, y)

vf = np.vectorize(f)
Z = vf(X, Y)


fig = plt.figure()
ax = fig.add_subplot(121)
ax.contour(X, Y, Z)
ax.plot(xhist, yhist, 'r-')

ax = fig.add_subplot(122, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)

zhist = [f(a, b) for a, b in zip(xhist, yhist)]
ax.plot(xhist, yhist, zhist, 'r-')
plt.show()
