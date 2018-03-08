""" Given f(x, y), dfx(x, y) and dfy(x, y), this file implements the steepest
    gradient descent algorithm for f(x, y)"""
from scipy.optimize import minimize_scalar


def f(x, y):
    # return x**2 + 10*(y - x**2)**2
    return 3*x**2 + y**2


def dfx(x, y):
    # return 2*(20*x**3 - 20*x*y + x)
    return 6*x


def dfy(x, y):
    # return 20*(y - x**2)
    return 2*y


class GFunction:
    """ GFunction turns @f into a single variable function of h
        after each descent. """
    def __init__(self, f, x, y):
        self.f = f
        self.x = x
        self.y = y
        self.dfx = dfx(x, y)
        self.dfy = dfy(x, y)

    def __call__(self, h):
        x = self.x - h * self.dfx
        y = self.y - h * self.dfy
        return self.f(x, y)


def calc_step_size(x, y):
    """ calculate step size h by argmin{g(h)}} """
    g = GFunction(f, x, y)
    ret = minimize_scalar(g)
    return ret.x, g.dfx, g.dfy


def steepest_gradient_descent(xk, yk):
    alpha, df_dx, df_dy = calc_step_size(xk, yk)

    delta_x, delta_y = alpha * df_dx, alpha * df_dy
    xk -= delta_x
    yk -= delta_y
    if abs(delta_x) <= 1e-6 and abs(delta_y) <= 1e-6:
        stop_hint = True
    else:
        stop_hint = False
    return xk, yk, stop_hint




