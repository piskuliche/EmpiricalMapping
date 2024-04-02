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

    @classmethod
    def build_from_map(cls, popt, xdata=None, ydata=None, xlabel="E", ylabel="w"):
        instance = cls.__new__(cls)
        instance.xdata = xdata
        instance.ydata = ydata
        instance.xlabel = xlabel
        instance.ylabel = ylabel

        instance.degree = len(popt) - 1
        instance.popt = popt

        instance.initial_guess = None
        instance.fit_bounds = (-np.inf, np.inf)
        instance.fit_keywords = {}
        return instance

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
        if self.xdata is None or self.ydata is None:
            raise ValueError(
                "This is a prebuilt map, you cannot fit it to a polynomial.")
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

    def calculate_rmse(self):
        """ Calculate the RMSE value of the fit.

        Returns:
        --------
        rmse: float
            The RMSE value of the fit.
        """
        if self.popt is None:
            raise ValueError(
                "You must fit the data to a polynomial before calculating the RMSE value.")
        return np.sqrt(np.sum(self.calculate_fit_error()**2) / (len(self.xdata)-len(self.popt)))

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
        if self.ydata is not None:
            print("R value: ", np.sqrt(self.calculate_r_squared()))
            print("R^2 value: ", self.calculate_r_squared())
            print("RMSE: ", self.calculate_rmse())
        print("*** ******** ***")
        return

    def report_latex(self):
        """ Print a report of the fit to the console in latex format.

        """
        if self.popt is None:
            raise ValueError(
                "You must fit the data to a polynomial before reporting the fit.")
        if len(self.popt) == 2:
            print(f"{self.ylabel} = {self.popt[0]:10.10f} + {self.popt[1]:10.10f} {self.xlabel} & " +
                  f"{np.sqrt(self.calculate_r_squared())} & " +
                  f"{self.calculate_rmse()}")
        else:
            print(f"{self.ylabel} = {self.popt[0]:10.10f} + {self.popt[1]:10.10f} {self.xlabel} + {self.popt[2]:10.10f} {self.xlabel}^2 & " +
                  f"{np.sqrt(self.calculate_r_squared())} & " +
                  f"{self.calculate_rmse()}")
        return

    @staticmethod
    def _format_scientific_latex(sci_number):
        import re
        if 'e' not in sci_number:
            return sci_number
        return re.sub(r"e(-)?(\d+)", r" \\times 10^{\1\2}", sci_number)

    def report_latex_line(self, yunit="\\textrm{cm}^{-1}", xunit='\\textrm{a.u.}', xlabel="E", popt_fmt=None, r_fmt="{:.2f}", rmse_fmt="{:.2f}"):
        """ This is a function for reporting a line in a latex table associated with this map.

        Parameters:
        -----------
        yunit: str
            The unit for the y-axis.
        xunit: str
            The unit for the x-axis.
        xlabel: str
            The label for the x-axis.
        popt_fmt: list
            The format for the optimal parameters.
        r_fmt: str
            The format for the R value.
        rmse_fmt: str
            The format for the RMSE value.

        Returns:
        --------
        line: str
            The latex line for the table.

        """
        if popt_fmt is None:
            popt_fmt = ["{:.2f}"]*len(self.popt)
        if len(popt_fmt) != len(self.popt):
            raise ValueError(
                "popt_fmt must be the same length as the number of optimal parameters.")
        if not isinstance(xlabel, str):
            xlabel = str(xlabel)

        # Strip the $ from the ylabel
        ylabel = self.ylabel.replace("$", "")
        if "~" in ylabel:
            ylabel = ylabel.split("~")[0]
        line_start = f"${ylabel} = "

        # Build the first column of the line
        popt_str = ""
        for i, p in enumerate(self.popt):
            fmt_p = popt_fmt[i].format(p)
            if "e" in fmt_p:  # formats in 10^{x} notation for latex
                fmt_p = self._format_scientific_latex(fmt_p)
            if i == 0:
                unit = f" \\left[{yunit}\\right] "
                popt_str += f"{fmt_p} {unit}"
            elif i == 1:
                unit = f" \\left[\\frac{{{yunit}}}{{{xunit}}}\\right] "
                popt_str += (
                    f"  {fmt_p} {unit} {xlabel}"
                    if "-" in fmt_p[0]
                    else f" + {fmt_p} {unit} {xlabel}"
                )
            else:
                unit = f" \\left[\\frac{{{yunit}}}{{{xunit}^{i}}}\\right] "
                popt_str += (
                    f"  {fmt_p} {unit} {xlabel}^{i}"
                    if "-" in fmt_p[0]
                    else f" + {fmt_p} {unit} {xlabel}^{i}"
                )
        popt_str += "$"

        r = np.sqrt(self.calculate_r_squared())
        r_str = r_fmt.format(r)
        if "e" in r_str:
            r_str = self._format_scientific_latex(r_str)

        rmse = self.calculate_rmse()
        rmse_str = rmse_fmt.format(rmse)
        if "e" in rmse_str:
            rmse_str = self._format_scientific_latex(rmse_str)

        line_end = "$ \\\\"
        return line_start + popt_str + " & $ " + r_str + " $ & $ " + rmse_str + line_end

    def display_map(self, xvals=None, **kwargs):
        # sourcery skip: class-extract-method
        """ Display the data and the fit on a plot.

        """
        if self.popt is None:
            raise ValueError(
                "You must fit the data to a polynomial before displaying the map.")
        fig = plt.figure(**kwargs)
        if self.xdata is not None:
            plt.scatter(self.xdata, self.ydata, c='black', label='data')
        if xvals is None and self.xdata is not None:
            xvals = np.linspace(min(self.xdata), max(self.xdata), 100)

        plt.plot(xvals, self.get_fit(xvals), c='red', label='fit')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend(frameon=False)
        plt.show()
        return

    def display_hist2d_map(self, xvals=None, bins=(100, 100), cmap='Blues', **kwargs):
        """ Display a 2D histogram of the data.

        Parameters:
        -----------
        bins: tuple
            The bins for the histogram.
        cmap: str or Colormap
            The colormap for the histogram.

        """
        if self.xdata is None or self.ydata is None:
            raise ValueError(
                "You must have xdata and ydata to display a 2D histogram.")
        if xvals is None:
            xvals = np.linspace(min(self.xdata), max(self.xdata), 100)

        fig = plt.figure(**kwargs)
        plt.hist2d(self.xdata, self.ydata, bins=bins, cmap=cmap, density=True)
        plt.plot(xvals, self.get_fit(xvals), c='red', label='fit')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.colorbar()
        plt.show()
        return

    def get_fit(self, xdata):
        """ Return the fit for a given xdata.

        Parameters:
        -----------
        xdata: array
            The xdata for which to return the fit.

        Returns:
        --------
        fit: array
            The fit for the given xdata.

        """
        if self.popt is None:
            raise ValueError(
                "You must fit the data to a polynomial before returning the fit.")
        return self._get_poly()(xdata, *self.popt)

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
        """ Initialize the FullMap object.
        """
        self.maps = {}
        self.order = {}

    def add_map(self, label, order,  xdata, ydata, **kwargs):
        """ Add a map to the FullMap object.

        Parameters:
        -----------
        label: str
            The label for the map.
        order: int
            The order of the polynomial to fit the map to.
        xdata: array
            The xdata for the map.
        ydata: array
            The ydata for the map.
        kwargs: dict
            Additional keyword arguments to be passed to the Map object.

        """

        self.maps[label] = Map(xdata, ydata, **kwargs)
        self.order[label] = order
        return

    def add_prebuilt_map(self, label, popt, **kwargs):
        """ Add a prebuilt map to the FullMap object.

        Parameters:
        -----------
        label: str
            The label for the map.
        popt: array
            The optimal parameters for the map.
        kwargs: dict
            Additional keyword arguments to be passed to the Map object.


        """
        self.maps[label] = Map.build_from_map(popt, **kwargs)
        self.order[label] = self.maps[label].degree
        return

    def fit_maps(self):
        """
        Fit all maps to their respective polynomials.
        """
        for label in self.maps.keys():
            self.maps[label].fit_to_poly(self.order[label])
        return

    def fit_map(self, label):
        """
        Fit a single map to its respective polynomial.
        """
        self._test_existence(label)
        self.maps[label].fit_to_poly(self.order[label])

    def report_maps(self, display=False, histdisplay=False, **kwargs):
        """
        Report all maps in the FullMap object.

        Parameters:
        -----------
        display: bool
            Whether to display the maps.
        kwargs: dict
            Additional keyword arguments to be passed to the display_map function. 
            This really supplies keyword arguments to plt.figure(), so dpi=x, figsize=(x,y) etc.

        """

        # Handle arguments
        bins = None
        cmap = None
        if 'bins' in kwargs:
            bins = kwargs['bins']
            del kwargs['bins']
        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
            del kwargs['cmap']

        for label in self.maps.keys():
            print(f"Map: {label}")
            self.maps[label].report_map()

            if display:
                self.maps[label].display_map(**kwargs)
            if histdisplay:
                if bins is None:
                    bins = (100, 100)
                if cmap is None:
                    cmap = 'Blues'
                self.maps[label].display_hist2d_map(
                    bins=bins, cmap=cmap, **kwargs)
        return

    def add_fit_guess(self, label, guess):
        """
        Add an initial guess to the fit.

        Parameters:
        -----------
        label: str
            The label for the map.
        guess: array
            The initial guess for the fit.

        """
        self._test_existence(label)
        self.maps[label].add_initial_guess(guess)
        return

    def add_fit_bounds(self, label, guess):
        """
        Add bounds to the fit.

        Parameters:
        -----------
        label: str
            The label for the map.
        guess: array
            The bounds for the fit.

        """
        self._test_existence(label)
        self.maps[label].add_bounds(guess)
        return

    def add_fit_keyword(self, label, keyword, value):
        """
        Add a keyword to the fit.

        Parameters:
        -----------
        label: str
            The label for the map.
        keyword: str
            The keyword for the fit.
        value: any
            The value for the keyword.
        """
        self._test_existence(label)
        self.maps[label].fit_keywords[keyword] = value
        return

    def report_latex_table(self):
        """ Print a latex table of the fits to the console. Note - uses default parameters.

        """
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\begin{tabular}{l c c}",
            "\\hline",
            "Map & Fit & RMSE \\\\",
            "\\hline",
        ]
        for map in self.maps:
            if 'x' in map:
                line = self.maps[map].report_latex_line(xlabel="\omega_{10}",
                                                        xunit="\\textrm{cm}^{-1}", yunit='\\AA', popt_fmt=["{:.4f}", "{:.4e}"], rmse_fmt="{:.4e}")
            elif "mu" in map:
                line = self.maps[map].report_latex_line(
                    xlabel="E", yunit="\\textrm{D}", rmse_fmt="{:.4e}")
            elif "iso" in map:
                line = self.maps[map].report_latex_line(
                    xlabel="E", yunit="\\textrm{a.u.}", rmse_fmt="{:.4e}")
            else:
                line = self.maps[map].report_latex_line(
                    xlabel="E", yunit="\\textrm{cm}^{-1}", rmse_fmt="{:.4e}")
            lines.append(line)
        lines.extend(("\\hline",
                      "\\end{tabular}",
                      "\\end{table}"))

        for line in lines:
            print(line)
        return

    def _test_existence(self, label):
        """
        Test if a map exists in the FullMap object.

        Parameters:
        -----------
        label: str
            The label for the map.

        Raises:
        -------
        ValueError:
            If the map does not exist.

        """
        if label not in self.maps.keys():
            raise ValueError(f"Map {label} not found.")
        return
