def poly_fit_selector(order):
    """
    Returns a list of fitting functions for the given order.

    Args:
        order: int, the order of the fitting function.

    Returns:
        fitfunc: function, the fitting function.
    """
    if order == 1:
        def fitfunc(x, a, b):
            return a + b*x
        return fitfunc
    elif order == 2:
        def fitfunc(x, a, b, c):
            dx = x - b
            return a + 0.5*c*dx**2
        return fitfunc
    elif order == 3:
        def fitfunc(x, a, b, c, d):
            dx = x - b
            return a + 0.5*c*dx**2 + d*dx**3
        return fitfunc
    elif order == 4:
        def fitfunc(x, a, b, c, d, e):
            dx = x - b
            return a + 0.5*c*dx**2 + d*dx**3 + e*dx**4
        return fitfunc
    elif order == 5:
        def fitfunc(x, a, b, c, d, e, f):
            dx = x - b
            return a + 0.5*c*dx**2 + d*dx**3 + e*dx**4 + f*dx**5
        return fitfunc
    elif order == 6:
        def fitfunc(x, a, b, c, d, e, f, g):
            dx = x - b
            return a + 0.5*c*dx**2 + d*dx**3 + e*dx**4 + f*dx**5 + g*dx**6
        return fitfunc
    elif order == 7:
        def fitfunc(x, a, b, c, d, e, f, g, h):
            dx = x - b
            return a + 0.5*c*dx**2 + d*dx**3 + e*dx**4 + f*dx**5 + g*dx**6 + h*dx**7
        return fitfunc
    elif order == 8:
        def fitfunc(x, a, b, c, d, e, f, g, h, i):
            dx = x - b
            return a + 0.5*c*dx**2 + d*dx**3 + e*dx**4 + f*dx**5 + g*dx**6 + h*dx**7 + i*dx**8
        return fitfunc
    elif order == 9:
        def fitfunc(x, a, b, c, d, e, f, g, h, i, j):
            dx = x - b
            return a + 0.5*c*dx**2 + d*dx**3 + e*dx**4 + f*dx**5 + g*dx**6 + h*dx**7 + i*dx**8 + j*dx**9
        return fitfunc
    elif order == 10:
        def fitfunc(x, a, b, c, d, e, f, g, h, i, j, k):
            dx = x - b
            return a + 0.5*c*dx**2 + d*dx**3 + e*dx**4 + f*dx**5 + g*dx**6 + h*dx**7 + i*dx**8 + j*dx**9 + k*dx**10
        return fitfunc


def mu_fit_selector(order):
    """
    Returns a list of fitting functions for the given order.

    Args:
        order: int, the order of the fitting function.

    Returns:
        mfunc: function, the fitting function.

    """
    if order == 1:
        def mfunc(x, a, b):
            return a + b*x
        return mfunc
    if order == 2:
        def mfunc(x, a, b, c):
            return a + b*x + c*x**2
        return mfunc
    if order == 3:
        def mfunc(x, a, b, c, d):
            return a + b*x + c*x**2 + d*x**3
        return mfunc
    if order == 4:
        def mfunc(x, a, b, c, d, e):
            return a + b*x + c*x**2 + d*x**3 + e*x**4
        return mfunc
    if order == 5:
        def mfunc(x, a, b, c, d, e, f):
            return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5
        return mfunc
    if order == 6:
        def mfunc(x, a, b, c, d, e, f, g):
            return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6
        return mfunc
    if order == 7:
        def mfunc(x, a, b, c, d, e, f, g, h):
            return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7
        return mfunc
