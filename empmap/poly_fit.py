""" This module contains functions to fit the potential energy and dipole moment to a polynomial. 

Notes:
------
These functions are used to fit the potential energy and dipole moment to a polynomial. The potential energy is fit to a polynomial of the form:
V = a + 0.5*c*dx^2 + d*dx**3 + e*dx**4 + ...

    Where dx = x - b.

The dipole moment is fit to a polynomial of the form:
mu = a + b*x + c* x^2 + ...
    
    Where x is the bond distance.

Examples:
---------
>>> from empmap.poly_fit import poly_fit_selector
>>> poly_fit_selector(2)
<function poly_fit_selector.<locals>.fitfunc at 0x7f9e8e9b7f28>

>>> from empmap.poly_fit import mu_fit_selector
>>> mu_fit_selector(2)
<function mu_fit_selector.<locals>.mfunc at 0x7f9e8e9b7f28>

"""

__all__ = ["poly_fit_selector", "mu_fit_selector"]


def poly_fit_selector(order):
    """
    Returns a list of fitting functions for the given order.

    Notes:
    ------
    These fitting functions are used to fit the potential energy to a polynomial.

    They are of the form:
    V = a + 0.5*c*dx^2 + d*dx**3 + e*dx**4 + ...

    Where dx = x - b.

    Special Case: If the order is 1, the fitting function is linear:
    V = a + b*x

    Parameters:
    ----------
        order: int
            The order of the fitting function.

    Returns:
    -------
        fitfunc: function
            The fitting function. Parameters are in order (x, a b, c, d, e, ...)

    Raises:
    ------
        ValueError: If the order is not an integer.

    """
    if not isinstance(order, int):
        raise ValueError("The order must be an integer.")

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

    Notes:
    ______
    These fitting functions are used to fit the dipole moment to a polynomial.

    They are of the form:
    mu = a + b*x + c* x^2 + ...

    Parameters:
    ----------
        order: int
            The order of the fitting function. Available options are 1, 2, 3, 4, 5, 6, 7.

    Returns:
    -------
        mfunc: function
            The fitting function. Parameters are in order (x, a, b, c, d, e, f, g)

    Raises:
    ------
        ValueError: If the order is not an integer.

    """
    if not isinstance(order, int):
        raise ValueError("The order must be an integer.")

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
