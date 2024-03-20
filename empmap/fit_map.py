from empmap.poly_fit import mu_fit_selector
from scipy.optimize import curve_fit
import numpy as np


class Map:
    def __init__(self, xdata, ydata, xlabel="E", ylabel="w"):
        self.xdata = xdata
        self.ydata = ydata
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.poly = None
        self.popt = None

    def fit_to_poly(self, degree, **kwargs):
        """ Fit the data to a polynomial of degree 'degree' and return the optimal parameters.

        Parameters:
        -----------
        degree: int
            The degree of the polynomial to fit the data to.
        kwargs: dict
            Additional keyword arguments to be passed to scipy.optimize.curve_fit.

        Returns:
        --------
        popt: array
            The optimal parameters for the polynomial fit."""
        self.poly = mu_fit_selector(degree)
        self.popt, self.pcov = curve_fit(
            self.poly, self.xdata, self.ydata, **kwargs)
        return

    def calculate_fit_error(self):
        """ Calculate the error between the data and the fit.

        Returns:
        --------
        error: array
            The error between the data and the fit.

        """
        if self.popt is None:
            raise ValueError(
                "You must fit the data to a polynomial before calculating the fit error.")
        return self.ydata - self.poly(self.xdata, *self.popt)

    def calculate_r_squared(self):
        """ Calculate the R^2 value of the fit.

        Returns:
        --------
        r_squared: float
            The R^2 value of the fit.

        """
        if self.popt is None:
            raise ValueError(
                "You must fit the data to a polynomial before calculating the R^2 value.")
        return 1 - (np.sum(self.calculate_fit_error()**2) / np.sum((self.ydata - np.mean(self.ydata))**2))

    def report_map(self):
        """ Print a report of the fit to the console.

        """
        if self.popt is None:
            raise ValueError(
                "You must fit the data to a polynomial before reporting the fit.")
        print(f"{self.ylabel} = {self.popt[0]:10.5f}" + [
              f" + {self.popt[i]:10.5f} * {self.xlabel}^{i}" for i in range(1, len(self.popt))] + ".")
        print("Optimal parameters: ", self.popt)
        print("R^2 value: ", self.calculate_r_squared())
        print("Error: ", self.calculate_fit_error())
        return
