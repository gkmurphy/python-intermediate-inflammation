#!/usr/bin/env python3
"""Software for managing and analysing patients' inflammation data in our imaginary hospital."""

import argparse

from inflammation import models, views

# This is me playing around
# alt + shift + E to run single line of code
# import package
import numpy as np

# pull in data and make variable and view
data = np.loadtxt(fname='data/inflammation-01.csv', delimiter=',')
data.shape

# Pull in only one func you want from partular file
from inflammation.models import daily_mean

daily_mean(data[0:4])

## UNIT TESTING
#Make fake array for testing - could be "pos integers" or "zeroes"
import numpy.testing as npt

#TEST 1
test_input = np.array([[1,2],[3,4],[5,6]])
test_result = np.array([3,4])
#Test that calculated result equals expected (same shape and elements)
npt.assert_array_equal(daily_mean(test_input), test_result)

#TEST 2
test_input = np.array([[2,0], [4,0]])
test_result = np.array([3,0])
npt.assert_array_equal(daily_mean(test_input), test_result)

#TEST 3
test_input = np.array([[0,0], [0,0], [0,0]])
test_result = np.array([0,0])
npt.assert_array_equal(daily_mean(test_input), test_result)

#TEST 4
test_input = np.array([[1,2], [3,4], [5,6]])
test_result = np.array([3,4])
npt.assert_array_equal(daily_mean(test_input), test_result)


def main(args):
    """The MVC Controller of the patient inflammation data system.

    The Controller is responsible for:
    - selecting the necessary models and views for the current task
    - passing data between models and views
    """
    infiles = args.infiles
    if not isinstance(infiles, list):
        infiles = [args.infiles]

    for filename in infiles:
        inflammation_data = models.load_csv(filename)
        # Add hanging indent below for readability ; open brackets make it clear what's inluded
        # between the brackets. Align like data within the brackets.
        view_data = {
            'average': models.daily_mean(inflammation_data),
            'max': models.daily_max(inflammation_data),
            'min': models.daily_min(inflammation_data)
        }

        views.visualize(view_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A basic patient inflammation data management system')

    parser.add_argument(
        'infiles',
        nargs='+',
        help='Input CSV(s) containing inflammation series for each patient')

    args = parser.parse_args()

    main(args)



def patient_normalise(data):
    """"Normalize patient data from a 2D inflammation data array"""
    #First find the max for each row then divide each value by the max for that row
    #newaxis converts it from a 1D to a 2D array
    #broadcasting allows Numpy to handle arrays of different shapes
    max = np.max(data, axis=1)
    return data / max[:, np.newaxis]

@py.test.mark.parameterize(
    "test, expected",
    [
        ([[1,2,3],[4,5,6],[7,8,9]], [[0.33, 0.67,1],[0.67,0.83,1],[0.78,0.89,1]])
        ])
def test_patient_normalise(test, expected):
    """Test normalisation works for arrays of one and positive integers.
    Assumption that test accuracy of tho decimal places is sufficient"""
    #Use almost equal if decimal places are arbitrary
    from inflammation.models import patient_normalise
    npt.assert_array_almost_equal(patient_normalise(np.array(test)),
    np.array(expected), decimal =2)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]]),
    ])

def patient_normalise(data):
    """
    Normalise patient data from a 2D inflammation data array.

    NaN values are ignored, and normalised to 0.

    Negative values are rounded to 0.
    """
    max = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised

def daily_above_threshold(patient_num, data, threshold):
    """Determine whether or not each daily inflammation value exceeds a given threshold for 
    a given patient.   
    :param patient_num: The patient row number
    :param data: A 2D data array with inflammation data
    :param threshold: An inflammation threshold to check each daily value against
    :returns: A boolean array representing whether or not each patient's daily inflammation
    exceeded the threshold
    """"

    return map(lambda x: x > threshold, data[patient_num])

from functools import reduce

    l = [1, 2, 3, 4]

def add(a, b):
    return a + b

    print(reduce(add, l))

    # The same reduction using a lambda function
    print(reduce((lambda a, b: a + b), l))