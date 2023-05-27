# Problem Set 4: Sea Level Rise
# Name: Iris Pang

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import scipy.stats as st
from scipy.interpolate import interp1d


def calculate_std(upper, mean):
    """
	Calculate standard deviation based on the upper 97.5th percentile

	Args:
		upper: a 1-d numpy array with length N, representing the 97.5th percentile
            values from N data points
		mean: a 1-d numpy array with length N, representing the mean values from
            the corresponding N data points

	Returns:
		a 1-d numpy array of length N, with the standard deviation corresponding
        to each value in upper and mean
	"""
    #calc standardized effect size, getting critical val from normal distr 
    return (upper - mean) / st.norm.ppf(.975) 

def load_data():
    """
	Loads data from sea_level_change.csv and puts it into numpy arrays

	Returns:
		a length 3 tuple of 1-d numpy arrays:
		    1. an array of years as ints
		    2. an array of 2.5th percentile sea level rises (as floats) for the years from the first array
		    3. an array of 97.5th percentile of sea level rises (as floats) for the years from the first array
        eg.
            (
                [2020, 2030, ..., 2100],
                [3.9, 4.1, ..., 5.4],
                [4.4, 4.8, ..., 10]
            )
            can be interpreted as:
                for the year 2020, the 2.5th percentile SLR is 3.9ft, and the 97.5th percentile would be 4.4ft.
	"""
    #read by comma separated values
    df = pd.read_csv('sea_level_change.csv')
    df.columns = ['Year', 'Lower', 'Upper']
    #convert to 1-D arrays
    return (df.Year.to_numpy(), df.Lower.to_numpy(), df.Upper.to_numpy())

def predicted_sea_level_rise(show_plot=False):
    """
	Creates a numpy array from the data in sea_level_change.csv where each row
    contains a year, the mean sea level rise for that year, the 2.5th percentile
    sea level rise for that year, the 97.5th percentile sea level rise for that
    year, and the standard deviation of the sea level rise for that year. If
    the year is between 2020 and 2100, inclusive, and not included in the data, 
    the values for that year should be interpolated. If show_plot, displays a 
    plot with mean and the 95% confidence interval, assuming sea level rise 
    follows a linear trend.

	Args:
		show_plot: displays desired plot if true

	Returns:
		a 2-d numpy array with each row containing the year, the mean, the 2.5th 
        percentile, 97.5th percentile, and standard deviation of the sea level rise
        for the years between 2020-2100 inclusive
	"""
    sea_data = load_data() #get data from csv file

    #from tuple, get respective arrays
    years = sea_data[0]
    lower = sea_data[1]
    upper = sea_data[2]

    #create interpolation functions from arrays
    f_slow = interp1d(years, lower)
    f_fast = interp1d(years, upper)
    
    #get new array of years from 2020 to 2100 incrememnted by 1 yr
    years_new = np.arange(2020, 2101)

    #interpolation of the slower and upper bounds
    slow_slr = f_slow(years_new)
    fast_slr = f_fast(years_new)

    #initialize new array of length 81 (to match prev arrays)
    mean_arr = np.zeros(81)

    #get mean using 2.5 and 97.5 percentiles
    for i in range(len(mean_arr)):
        mean_arr[i] = (slow_slr[i] + fast_slr[i])/2

    #calculate std
    std_arr = calculate_std(fast_slr, mean_arr)

    #stack all arrays together for one large 2-D array
    slr_data = np.column_stack((years_new, mean_arr, slow_slr, fast_slr, std_arr))

    if show_plot:
        #plot the lines
        plt.plot(years_new, mean_arr, color = "green", label = "Mean")
        plt.plot(years_new, slow_slr, color = "orange", linestyle = "dashed", label = "Lower bound")
        plt.plot(years_new, fast_slr, color = "blue", linestyle = "dashed", label = "Upper bound")


        plt.xlabel('Year')
        plt.ylabel('Projected annual mean water level (ft)')
        plt.legend()
        plt.title("Simulated projected sea level change in feet from 2020 to 2100")
    
        plt.show()

    return slr_data



def simulate_year(data, year, num):
    """
	Simulates the sea level rise for a particular year based on that year's
    mean and standard deviation, assuming a normal distribution. 

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
		year: the year to simulate sea level rise for
        num: the number of samples you want from this year

	Returns:
		a 1-d numpy array of length num, that contains num simulated values for
        sea level rise during the year specified
	"""
    #counter for year to traverse along 2-D arrays
    i = year - 2020
    #get the year and mean/std
    mean = data[i][1]
    std = data[i][-1]

    #get a random sample from normal distribution
    sample = np.random.normal(mean, std, num)

    return sample
    
    
def plot_simulation(data):
    """
	Runs and plots a Monte Carlo simulation, based on the values in data and
    assuming a normal distribution. Five hundred samples should be generated
    for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
	"""
    #get column of years
    years = data[:,0]
    
    #get the samples from normal distributions, with 500 reps
    sample_arr = []
    for i in range(81):
        sample = simulate_year(data, i+2020, 500)
        sample_arr.append(sample)

    #plot the mean, lower and upper bounds
    plt.plot(data[:,0], data[:,1], color = "green", label = "Mean")
    plt.plot(data[:,0], data[:,2], color = "orange", linestyle = "dashed", label = "Lower bound")
    plt.plot(data[:,0], data[:,3], color = "blue", linestyle = "dashed", label = "Upper bound")

    #plot the different samples
    for i in range(81):
        xarr = np.full(500, years[i]) #make a sep array for years to have shape of 500
        plt.scatter(
            xarr, sample_arr[i], c="gray", s=0.1)
        
    plt.xlabel('Year')
    plt.ylabel('Projected annual mean water level (ft)')
    plt.legend()
    plt.title("Simulated projected sea level change in feet from 2020 to 2100")
    plt.show()
    
def simulate_water_levels(data):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year

	Returns:
		a python list of simulated water levels for each year, in the order in which
        they would occur temporally
	"""
    year_list = []

    #traverse accordingly to index that corresponds to year
    for i in range(81):
        #get simulated data point 
        year = simulate_year(data, i+2020, 1)
        #converting array to list
        year_list.append(float(year[0]))
    return year_list
        


def repair_only(water_level_list, water_level_loss_no_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a repair only strategy, where you would only pay
    to repair damage that already happened.

    The repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft (exclusive), the cost is the
           house_value times the percentage of property damage for that water
           level. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the first column is
            the SLR levels and the second column is the corresponding property damage expected
            from that water level with no flood prevention (as an integer percentage)
        house_value: the value of the property we are estimating cost for

	Returns:
		a python list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    damage_cost = []

    for yr in water_level_list:
        #if less than 5 ft, no cost
        if yr <= 5:
            damage_cost.append(0)
        #if eq to or more than 10, whole cost (in Ks)
        elif yr >= 10:
            damage_cost.append(house_value/1000.0)
        #interpolate to get percent to multiply the house value by
        else:
            proppercent = interp1d(water_level_loss_no_prevention[:,0], water_level_loss_no_prevention[:,1])
            damagepercent = proppercent(yr)/100.0
            damage_cost.append(damagepercent*house_value/1000.0)

    return damage_cost





def wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000,
               cost_threshold=100000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a wait a bit to repair strategy, where you start
    flood prevention measures after having a year with an excessive amount of
    damage cost.

    Flood prevention measures are put into place if you have any year with a
    damage cost above the cost_threshold.

    The wait a bit to repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft (exclusive), the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    
    damage_cost = []
    preparation_taken = False

    for yr in water_level_list:
        #if less than 5 ft, no cost
        if yr <= 5:
            damage_cost.append(0)
         #if eq to or more than 10, whole cost (in Ks)
        elif yr >= 10:
            damage_cost.append(house_value/1000.0)
        else:
            #get the damage done to see if prep needs to be taken
            proppercent = interp1d(water_level_loss_no_prevention[:,0], water_level_loss_no_prevention[:,1])
            damagepercent = proppercent(yr)/100.0
            damage = damagepercent*house_value

            #for the first time an extreme amnt of money was used
            if damage > cost_threshold and not preparation_taken:
                preparation_taken = True
                damage_cost.append(damage/1000.0)
            #if preparation is already, taken interpolate with percentatages with prevention
            elif preparation_taken:
                protectcent = interp1d(water_level_loss_with_prevention[:,0], water_level_loss_with_prevention[:,1])
                damageprot = protectcent(yr)/100.0
                damage_cost.append(damageprot*house_value/1000.0)    
            #if not, tkae cost with interpolation with percentages w/o prevention
            else:
                damage_cost.append(damage/1000.0)
    return damage_cost


def prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a prepare immediately strategy, where you start
    flood prevention measures immediately.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_with_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The prepare immediately strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft (exclusive), the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    damage_cost = []

    for yr in water_level_list:
        #if less than 5 ft, no cost
        if yr <= 5:
            damage_cost.append(0)
        #if eq to or more than 10, whole cost (in Ks)
        elif yr >= 10:
            damage_cost.append(house_value/1000.0)
        #find costs by interpolation with costs with prevention
        else:
            proppercent = interp1d(water_level_loss_with_prevention[:,0], water_level_loss_with_prevention[:,1])
            damagepercent = proppercent(yr)/100.0
            damage_cost.append(damagepercent*house_value/1000.0)

    return damage_cost


def plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000,
                    cost_threshold=100000):
    """
	Runs and plots a Monte Carlo simulation of all of the different preparation
    strategies, based on the values in data and assuming a normal distribution.
    Five hundred samples should be generated for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, the 2.5th percentile, 97.5th percentile, mean, and standard
            deviation of the sea level rise for the given year
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place
	"""
    #get array of years
    years = data[:, 0]

    #initialize arrays of shape 81 to hold averages
    totrepair = np.zeros(81)
    totwait = np.zeros(81)
    totimmediate = np.zeros(81)

    #run 500 simulations over 81 years
    for sim in range(500):
        #get water levels
        water_lvl = simulate_water_levels(data)
        #get the lists representing costs over 81 years
        costrepair = repair_only(water_lvl, water_level_loss_no_prevention, house_value)
        costwait = wait_a_bit(water_lvl, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value, cost_threshold)
        costimmediate = prepare_immediately(water_lvl, water_level_loss_with_prevention, house_value)

        #plot scatter plots of these values for one simulation
        plt.scatter(years, costrepair, s=0.01, c="green")
        plt.scatter(years, costwait, s=0.01, c="blue")
        plt.scatter(years, costimmediate, s=0.01, c="red")

        #add the values to averages for one simulation (divided by 500 b/c eventually divide by 500)
        for y in range(81):
            totrepair[y] += costrepair[y]/500.0
            totwait[y] += costwait[y]/500.0
            totimmediate[y] += costimmediate[y]/500.0

    #plot averages
    plt.plot(years, totrepair, c="green", label = "Repair only scenario")
    plt.plot(years, totwait, c = "blue", label = "Wait-a-but scenario")
    plt.plot(years, totimmediate, c="red", label = "Prepare immediately scenario")

    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Estimated Damage Cost ($K)") 
    plt.title("Property damage cost comparison")
    plt.show()




if __name__ == '__main__':
    #pass

    data = predicted_sea_level_rise(show_plot=True)
    water_level_loss_no_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 10, 25, 45, 75, 100]]).T
    water_level_loss_with_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 5, 15, 30, 70, 100]]).T
    plot_simulation(data)
    print(wait_a_bit(simulate_water_levels(data), water_level_loss_no_prevention, water_level_loss_with_prevention, 400000, 100000))
    plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention)