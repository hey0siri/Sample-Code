# Problem Set 5: Modeling Temperature Change
# Name: Iris Pang

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAIN_INTERVAL = range(1961, 2000)
TEST_INTERVAL = range(2000, 2017)


def standard_error_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    #testing lengths of x and y arrays
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    
    #differences between actual and predicted y vals
    EE = ((estimated - y)**2).sum()
    #differences between actual and predicted x vals
    var_x = ((x - x.mean())**2).sum()

    #calculating standard error
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]


class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        #opening file to read
        f = open(filename, 'r')
        #getting headers of cities, temp, date
        header = f.readline().strip().split(',')


        for line in f:
            #split rows along data (city, temp, date)
            items = line.strip().split(',')

            #match each item to date pattern (YYYY/MM/DD) to items in DATE header
            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            #match along year, month, day
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            #collect city names under city header
            city = items[header.index('CITY')]
            #collect temperature floats under temp header
            temperature = float(items[header.index('TEMP')])
            
            #init data if city, year, or month not in raw data
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_daily_temps(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []

        #testing if city/year in data
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        #traversing along months and days in specified year
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]: #get the temperatures for each day in each month
                    temperatures.append(self.rawdata[city][year][month][day]) #add to array of temps
        return np.array(temperatures)

    def get_temp_on_date(self, city, month, day, year):
        """
        Get the temperature for the given city at the specified date.

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified date and city
        """
        #check if city, year, month, day in data
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        #return corr temp if available
        return self.rawdata[city][year][month][day]

    def calculate_annual_temp_averages(self, cities, years):
        """
        For each year in the given range of years, computes the average of the
        annual temperatures in the given cities.

        Args:
            cities: a list of the names of cities to include in the average
                annual temperature calculation
            years: a list of years to evaluate the average annual temperatures at

        Returns:
            a 1-d numpy array of floats with length = len(years). Each element in
            this array corresponds to the average annual temperature over the given
            cities for a given year.
        """
        averages = []

        #gor every year, calculate average
        for yr in years:
            total = [] #get total daily temperatures over one year
            for city in cities: 
                city_temp = self.get_daily_temps(city, yr) #get all the temp of a city over a yr
                total.extend(city_temp.tolist()) #add to total of temperatures
            averages.append(np.average(total)) #calculate the average of all these temperatures
        
        return np.array(averages)


def linear_regression(x, y):
    """
    Calculates a linear regression model for the set of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        (m, b): A tuple containing the slope and y-intercept of the regression line,
                both of which are floats.
    """

    x_avg = sum(x)/len(x) #get the avg of x vals
    y_avg = sum(y)/len(y) #ge the avg of y vals

    numerator = 0 #calculate the num/denom separately
    denominator = 0

    for xi, yi in zip(x,y): #look at the x and corresponding y at the same time
        numerator += (xi-x_avg)*(yi-y_avg) #calc num/denom according to eq
        denominator += (xi-x_avg)**2

    m = numerator/denominator #get m thru equation of SE
    b = y_avg - (m*x_avg) #get b thru equation of SE

    return (m, b)

def squared_error(x, y, m, b):
    """
    Calculates the squared error of the linear regression model given the set
    of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        m: The slope of the regression line
        b: The y-intercept of the regression line


    Returns:
        a float for the total squared error of the regression evaluated on the
        data set
    """
    #sum of the squared differences of the actual value to predicted value by linear regression
    return float(sum((yi - m*xi -b)**2 for xi, yi in zip(x,y))) 

def generate_polynomial_models(x, y, degrees):
    """
    Generates a list of polynomial regression models with degrees specified by
    degrees for the given set of data points

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        degrees: a list of integers that correspond to the degree of each polynomial
            model that will be fit to the data

    Returns:
        a list of numpy arrays, where each array is a 1-d numpy array of coefficients
        that minimizes the squared error of the fitting polynomial
        
        The models should appear in the list in the same order as their corresponding 
        integers in the `degrees` parameter
    """
    square_error = []
    #get array of coefficients for each fit for each degree (using numpy's polyfit)
    for deg in degrees:
        square_error.append(np.polyfit(x, y, deg))

    return square_error


def evaluate_models(x, y, models, display_graphs=False):
    """
    For each regression model, compute the R-squared value for this model and
    if display_graphs is True, plot the data along with the best fit curve. You should make a separate plot for each model.
   
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the R-squared value for each model
    """
    r2_list = []

    #for each polynomial mdoel
    for model in models:
        #get the function used to calc predicted y from the mdoel
        y_predicted = np.poly1d(model)
        y_p = []

        #get all the predicted y values as predicted by the function
        for xp in x:
            y_p.append(y_predicted(xp))

        #get the r^2 value from r2_score between actual and predicted y values
        r2val = r2_score(y, y_p)
        r2_list.append(r2val) #add to list of r^2 values for each model

        #convert to arrays for graphing
        y_p = np.array(y_p)
        x = np.array(x)

        if display_graphs:
            #if the graph is linear (model will have 2 coefficients, m and b if linear)
            if len(model) == 2:
                #get the standard error
                se = round(standard_error_over_slope(x, y, y_p, model),4)
                #init. title with respective values
                plt.title("R^2 = " + str(round(r2val, 4)) + "\n degree = " + str(len(model)-1) + "\n standard error over slope = " + str(se), loc = "center")
            else:
                plt.title("R^2 = " + str(round(r2val, 4)) + "\n degree = " + str(len(model)-1), loc = "center")
            #plot data points as individual points
            plt.scatter(x, y, c = "C0", label = "Data points")
            #plot the model as a solid line
            plt.plot(x, y_p, c = "C1", label = "Model")
            #Label axes
            plt.ylabel("Temperature in degrees Celsius") 
            plt.xlabel("Years")

            plt.legend()
            plt.show()

    return r2_list


def get_max_trend(x, y, length, positive_slope):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        length: the length of the interval
        positive_slope: a boolean whose value specifies whether to look for
            an interval with the most extreme positive slope (True) or the most
            extreme negative slope (False)

    Returns:
        a tuple of the form (i, j, m) such that the application of linear (deg=1)
        regression to the data in x[i:j], y[i:j] produces the most extreme
        slope m, with the sign specified by positive_slope and j-i = length.

        In the case of a tie, it returns the first interval. For example,
        if the intervals (2,5) and (8,11) both have slope 3.1, (2,5,3.1) should be returned.

        If no intervals matching the length and sign specified by positive_slope
        exist in the dataset then return None
    """
    maxslope = 0 #initialize a variable of maximumm slope

    intervals = {} #set up tuple of intervals that lead to slopes

    if len(x) < length: #if no intervals exist, return None
        return None
    
    #counter to splice the arrays, look thru each at certain intervals
    for i in range(len(x)-length + 1):
        #splice x and y accordingly to the interval length
        x_val = x[i:i+length]
        y_val = y[i:i+length]

        #get the m and b for these intervals' linear regressions
        coefficients = linear_regression(np.array(x_val), np.array(y_val))


        if positive_slope: #if asking for positive slope
            if coefficients[0] > maxslope + 10**-8: #if m is larger than maxslope within the tolerance
                maxslope = coefficients[0] #set maxslope and intervals
                intervals = (i, i+length)
        else: #if asking for negative slope
            if coefficients[0] < maxslope - 10**-8: #if m is less than maxslope within tolerance
                maxslope = coefficients[0] #set minslope and intervals
                intervals = (i, i+length)

    if positive_slope and maxslope <= 0:
        return None #if positive slope is negative
    if not positive_slope and maxslope >= 0:
        return None #if negative slope is positive

    return (intervals[0], intervals[1], maxslope) #get the intervals from tuple


def get_all_max_trends(x, y):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        a list of tuples of the form (i,j,m) such that the application of linear
        regression to the data in x[i:j], y[i:j] produces the most extreme
        positive OR negative slope m, and j-i=length.

        If len(x) < 2, return an empty list
    """
    slopelist = [] #initialize list of extreme tuples

    if len(x) < 2: #if the length of x is less than 2
        return []
    else:
        for interval in range(2, len(x)+2): #between 2 and len(x) + 1 (to set up scrolling through array)
            max_slope = get_max_trend(x, y, interval, True) #get min and max slope
            min_slope = get_max_trend(x, y, interval, False)

            #if max slope is negative/min slope is positive, an but interval is still valid (w/in len(x))
            if not max_slope and not min_slope and interval <= len(x): 
                slopelist.append((0, 0+interval, None))

            #if max_slope is positive but min_slope is positive (None)
            if max_slope and not min_slope:
                slopelist.append(max_slope) #add max slope no matter what
            #if min_slope is negative but max_slope is negative (None)
            elif min_slope and not max_slope:
                slopelist.append(min_slope) #add min slope no matter what
            elif min_slope and max_slope: #if both valid values
                if abs(min_slope[-1]) >= max_slope[-1]: #get the larger absolute value
                    slopelist.append(min_slope) #adding whole tuple
                else:
                    slopelist.append(max_slope)

    return slopelist


def calculate_rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    #calculate the numerator accordingly to RMSE equation
    numerator = sum(float((yi-estimatedi)**2) for yi, estimatedi in zip(y, estimated))
    #divide num by length of y (number of data points) and take square root
    return (numerator/len(y))**(1/2)


def evaluate_rmse(x, y, models, display_graphs=False):
    """
    For each regression model, compute the RMSE for this model and if
    display_graphs is True, plot the test data along with the model's estimation.

    RMSE rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N test data sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N test data sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the RMSE value for each model
    """

    rmse_list = [] #initialize list of RMSE values

    for model in models: #for every single model
        y_predicted = np.poly1d(model) #get function that calculates the y values
        y_p = []

        #ge the predicted y values for every single x value
        for xp in x:
            y_p.append(y_predicted(xp))

        #calculate rmse value
        rmseval = calculate_rmse(y, y_p)
        rmse_list.append(rmseval)

        #convert to arrays for plotting
        y_p = np.array(y_p)
        x = np.array(x)

        #for plotting
        if display_graphs:
            #title with needed information (RMSE and degree)
            plt.title("RMSE = " + str(round(rmseval, 4)) + "\n degree = " + str(len(model)-1), loc = "center")
            #plot data points as scatter plot pts
            plt.scatter(x, y, c = "Blue", label = "Data points")
            #plot model as solid line
            plt.plot(x, y_p, c = "Red", label = "Model")

            #labeling axes
            plt.ylabel("Temperature in degrees Celsius") 
            plt.xlabel("Years")

            plt.legend()
            plt.show()

    return rmse_list


if __name__ == '__main__':
    ##################################################################################
    # Implementation for trends along 1961-2017 in San Francisco for 12/01

    #getting the data from the csv file
    data_annual = Dataset("data.csv")

    #get the list of desired years
    x_val = []
    x_val.extend(range(1961,2017))

    #get the temperatures on the specific date of 12/01 for particular year
    y_val = []
    for yr in x_val:
        y_val.append(data_annual.get_temp_on_date("SAN FRANCISCO", 12, 1, yr))

    #convert x_val and y_val to an array
    x = np.array(x_val)
    y = np.array(y_val)

    #generate linear models
    modelsdaily = generate_polynomial_models(x, y,[1])

    #get r^2 values and plot
    evaluate_models(x, y, modelsdaily, True)


    ##################################################################################
    # Implementation for trends along average temperatures along 1961-2017 in San Francisco

    #calculate the annual averages over the same yr range for San Francisco
    y_avg = data_annual.calculate_annual_temp_averages(["SAN FRANCISCO"], x_val)
    #convert to an array
    y_arr_avg = np.array(y_avg)
    
    #generate the linear model for these values
    modelsann = generate_polynomial_models(x, y_arr_avg, [1])
    #evaluate r^2 values and plot
    evaluate_models(x, y_arr_avg, modelsann, True)

    ##################################################################################
    # Identifying 30 year window for extreme temperature increase in Seattle
    
    #get the annual temperatures of seattle across same yrs
    seattle_temp = data_annual.calculate_annual_temp_averages(["SEATTLE"], x_val)

    #get the max slope and interval to splice years and temp values
    (start, end, slope) = get_max_trend(x, seattle_temp, 30, True)

    #get intervals of years and temps
    max_years = x[start:end]
    max_temp = seattle_temp[start:end]

    #get m and b for a linear fit from data
    (slope1, y_int) = linear_regression(max_years, max_temp)

    #evaluate r^2 and plot
    evaluate_models(max_years, max_temp, [(slope, y_int)], True)


    ##################################################################################
    # Identifying 12 year window  for extreme temperature decrease in Seattle

    #get min (negative) slope and interval to splice years and temp values
    (start, end, slope) = get_max_trend(x, seattle_temp, 12, False)

    #get intervals of years and temps
    min_years = x[start:end]
    min_temp = seattle_temp[start:end]

    #get m and b for a linear fit from data
    (slope2, y_int) = linear_regression(min_years, min_temp)

    #evaluate r^2 and plot
    evaluate_models(min_years, min_temp, [(slope, y_int)], True)

    ##################################################################################
    # Generation of models based on training set of years

    #get the years for training set
    train_yr = []
    train_yr.extend(TRAIN_INTERVAL)

    #get the averages over the training years for all cities
    avg_temp = data_annual.calculate_annual_temp_averages(CITIES, train_yr)
    #get the 2nd and 10th degree polynomial models
    models = generate_polynomial_models(np.array(train_yr), avg_temp, [2, 10])
    #get r^2 values and plot
    evaluate_models(np.array(train_yr), avg_temp, models, True)

    ####################################################################################
    #Testing models on test set of years

    #get the years for testing set
    test_yr = []
    test_yr.extend(TEST_INTERVAL)

    #get averages over the testing years for all ciites
    testavg = data_annual.calculate_annual_temp_averages(CITIES, test_yr)
    #use the same 2nd and 10th degree models above to evaluate rmse
    evaluate_rmse(np.array(test_yr), testavg, models, True)