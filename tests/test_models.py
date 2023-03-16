"""Tests for statistics functions within the Model layer."""
import numpy as np
import numpy.testing as npt
import pytest


#We can also test for expected errors
def test_daily_min_string():
    '''Test for TypeError when passing strings'''
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hi', 'Hello'], ['Obi', 'Baby']])
def test_daily_max_negpos():
    """Test that the max function works for a mix of pos and neg integers"""
    from inflammation.models import daily_max

    test_input = np.array([[1,-1], [18,-18], [-17, 17], [6,0], [0,6]])
    test_result = np.array([18,17])
    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_max_dec():
    """Test that the max function works for a mix of integers and decimals"""
    from inflammation.models import daily_max

    test_input = np.array([[0,.1], [5.5, 9], [5.4, 8.9], [5.6, 9.2], [0.1, 10]])
    test_result = np.array([5.6, 10])
    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_min_negpos():
    """Test that the min function works for a mix of pos and neg integers"""
    from inflammation.models import daily_min

    test_input = np.array([[1, 1], [18, 0], [-17, 17], [6, 0], [0, 6]])
    test_result = np.array([-17, 0])
    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_dec():
    """Test that the max function works for a mix of pos and neg integers"""
    from inflammation.models import daily_min

    test_input = np.array([[1.3, -1], [18, -18.98], [17, 17], [6, 0], [0, 6]])
    test_result = np.array([0, -18.98])
    npt.assert_array_equal(daily_min(test_input), test_result)

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

##what if we tried to parameterize it 0.o
# parameterize is a python decorator which gives add'l functionality
#https://www.tutorialspoint.com/pytest/pytest_parameterizing_tests.htm#:~:text=Here%20the%20test%20multiplies%20an,other%20is%20the%20expected%20result.
import pytest
@pytest.mark.parametrize(
    "test, expected",
    [
        #Here we're putting the array and result in a single line
        ([ [0,0],[0,0],[0,0] ],[0,0]),
        ([ [1,2],[3,4],[5,6] ],[3,4]),
    ])
def test_daily_mean(test,expected):
    """Test mean function works for array of zeroes and pos integers"""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ], [0, 0, 0]),
        ([ [4, 2, 5], [1, 6, 2], [4, 1, 9] ], [4, 6, 9]),
        ([ [4, -2, 5], [1, -6, 2], [-4, -1, 9] ], [4, -1, 9]),
    ])
def test_daily_max(test, expected):
    """Test max function works for zeroes, positive integers, mix of positive/negative integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ], [0, 0, 0]),
        ([ [4, 2, 5], [1, 6, 2], [4, 1, 9] ], [1, 1, 2]),
        ([ [4, -2, 5], [1, -6, 2], [-4, -1, 9] ], [-4, -6, 2]),
    ])
def test_daily_min(test, expected):
    """Test min function works for zeroes, positive integers, mix of positive/negative integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))

## RANDOM DATA TESTING
import random

#seed sets a new range of random #s then you specify range
#were saving the randomness tho so it can be reused and tested
random.seed(1)
print(random.sample(range(0,100),10))
random.seed(1)
print(random.sample(range(0,100),10))

