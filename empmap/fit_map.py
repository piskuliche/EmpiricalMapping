from empmap.poly_fit import mu_fit_selector
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


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
        popt = self.popt
        if len(self.popt) == 2:
            popt = np.append(popt, 0)
        print(
            f"{self.ylabel} = {popt[0]:10.5f} + {popt[1]:10.5f} * {self.xlabel} + {popt[2]:10.5f} * {self.xlabel}^2")

        print("Optimal parameters: ", self.popt)
        print("R^2 value: ", self.calculate_r_squared())
        print("RMSE: ", np.sqrt(np.sum(self.calculate_fit_error()**2)) /
              (len(self.xdata)-len(self.popt)))
        return

    def display_map(self):
        """ Display the data and the fit on a plot.

        """
        if self.popt is None:
            raise ValueError(
                "You must fit the data to a polynomial before displaying the map.")
        plt.scatter(self.xdata, self.ydata, c='black', label='data')
        xvals = np.linspace(min(self.xdata), max(self.xdata), 100)
        plt.plot(xvals, self.poly(xvals, *self.popt), c='red',
                 label='fit')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.show()
        return


class FullMap:
    def __init__(self):
        self.maps = {}
        self.order = {}

    def add_map(self, label, order,  xdata, ydata, **kwargs):
        self.maps[label] = Map(xdata, ydata, **kwargs)
        self.order[label] = order
        return

    def fit_maps(self):
        for label in self.maps.keys():
            self.maps[label].fit_to_poly(self.order[label])
        return

    def report_maps(self, display=False):
        for label in self.maps.keys():
            print(f"Map: {label}")
            self.maps[label].report_map()
            if display:
                self.maps[label].display_map()
        return
