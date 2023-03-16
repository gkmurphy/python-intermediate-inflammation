"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt

import pytest

#We can also test for expected errors
def test_daily_min_string():
    '''Test for TypeError when passing strings'''
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hi', 'Hello'], ['Obi', 'Baby', 'Girl']])

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



@pytest.mark.parametrize(
    "test,expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[1, 0, 100], [2, 0, 1], [3, 0, 50]], [0.82, 0, 40.41]),

    ])
def test_dayly_std(test, expected):

    from inflammation.models import daily_std_dev

    npt.assert_array_almost_equal(daily_std_dev(np.array(test)), np.array(expected), decimal=2)