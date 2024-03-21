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

        self.degree = None
        self.popt = None
        self.initial_guess = None
        self.fit_bounds = (-np.inf, np.inf)
        self.fit_keywords = {}

    def fit_to_poly(self, degree):
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
        self.degree = degree
        self.popt, self.pcov = curve_fit(
            self._get_poly(), self.xdata, self.ydata, **self.fit_keywords)
        return

    def add_bounds(self, bounds):
        """ Add bounds to the fit.

        Parameters:
        -----------
        bounds: array
            The bounds for the fit.

        """
        if bounds is not None:
            # Check if bounds are a tuple
            if not isinstance(bounds, tuple):
                raise TypeError(f"Bounds must be a tuple. Got {type(bounds)}.")
            # Check if bounds are the correct length
            if len(bounds) != 2:
                raise ValueError(
                    f"Bounds must be of length 2. Got len({bounds}).")

        if bounds is None:
            bounds = (-np.inf, np.inf)
        self.fit_keywords['bounds'] = bounds
        return

    def add_initial_guess(self, initial_guess):
        """ Add an initial guess to the fit.

        Parameters:
        -----------
        initial_guess: array
            The initial guess for the fit.

        """
        if initial_guess is not None and not isinstance(initial_guess, list):
            raise TypeError(
                f"Initial guess must be a list. Got {type(initial_guess)}.")
        self.add_fit_keyword('p0', initial_guess)
        return

    def add_fit_keyword(self, keyword, value):
        """ Add a keyword to the fit.

        Parameters:
        -----------
        keyword: str
            The keyword for the fit.
        value: any
            The value for the keyword.

        """
        self.fit_keywords[keyword] = value
        return

    def remove_fit_keyword(self, keyword):
        """ Remove a keyword from the fit.

        Parameters:
        -----------
        keyword: str
            The keyword to remove from the fit.

        """
        if keyword in self.fit_keywords.keys():
            del self.fit_keywords[keyword]
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
        return self.ydata - self._get_poly()(self.xdata, *self.popt)

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
        print("*** ******** ***")
        print(f" Fit Report - {self.ylabel} vs {self.xlabel}")
        print("*** ******** ***")
        popt = self.popt
        if len(self.popt) == 2:
            popt = np.append(popt, 0)
        print(
            f"{self.ylabel} = {popt[0]:10.10f} + {popt[1]:10.10f} * {self.xlabel} + {popt[2]:10.10f} * {self.xlabel}^2")

        print("Optimal parameters: ", self.popt)
        print("R^2 value: ", self.calculate_r_squared())
        print("RMSE: ", np.sqrt(np.sum(self.calculate_fit_error()**2)) /
              (len(self.xdata)-len(self.popt)))
        print("*** ******** ***")
        return

    def display_map(self):
        """ Display the data and the fit on a plot.

        """
        if self.popt is None:
            raise ValueError(
                "You must fit the data to a polynomial before displaying the map.")
        plt.scatter(self.xdata, self.ydata, c='black', label='data')
        xvals = np.linspace(min(self.xdata), max(self.xdata), 100)
        plt.plot(xvals, self._get_poly()(xvals, *self.popt), c='red',
                 label='fit')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.show()
        return

    def _get_poly(self):
        """ Return the polynomial function for the fit.

        Returns:
        --------
        poly: function
            The polynomial function for the fit.

        """
        return mu_fit_selector(self.degree)


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

    def fit_map(self, label):
        self._test_existence(label)
        self.maps[label].fit_to_poly(self.order[label])

    def report_maps(self, display=False):
        for label in self.maps.keys():
            print(f"Map: {label}")
            self.maps[label].report_map()
            if display:
                self.maps[label].display_map()
        return

    def add_fit_guess(self, label, guess):
        self._test_existence(label)
        self.maps[label].add_initial_guess(guess)
        return

    def add_fit_bounds(self, label, guess):
        self._test_existence(label)
        self.maps[label].add_bounds(guess)
        return

    def add_fit_keyword(self, label, keyword, value):
        self._test_existence(label)
        self.maps[label].fit_keywords.update(keyword)

    def _test_existence(self, label):
        if label not in self.maps.keys():
            raise ValueError(f"Map {label} not found.")
        return
