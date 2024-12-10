"""
Name: test_trees.py
Author: John Yannotty (yannotty.1@osu.edu)
Desc: Test suite for trees.py

Start Date: 04/21/22
Version: 1.0

"""
# Imports
import os
import sys
import numpy as np

from pathlib import Path

from openbtmixing.tests.polynomial_models import sin_cos_exp
from openbtmixing import Openbtmix


_TEST_DATA = Path(__file__).parent.joinpath("bart_bmm_test_data").resolve()


# ---------------------------------------------
# Define the test functions
# ---------------------------------------------
# Test the mixing fun
def test_mixing():
    x_train = np.loadtxt(_TEST_DATA.joinpath('2d_x_train.txt')).reshape(80, 2)
    x_train = x_train.reshape(2, 80).transpose()

    y_train = np.loadtxt(_TEST_DATA.joinpath('2d_y_train.txt')).reshape(80, 1)

    f_train = np.concatenate([f1.evaluate(x_train)[0],f2.evaluate(x_train)[0]], axis = 1)

    # Set prior information
    sighat = 0.01/np.sqrt(7/5)
    mix.set_prior(
        k=2.5,
        ntree=30,
        nu=5,
        sighat=sighat,
        inform_prior=False)

    # Check tuning & hyper parameters
    assert mix.k == 2.5, "class object k is not set."
    assert mix.ntree == 30, "class object ntree is not set."
    assert mix.nu == 5, "class object nu is not set."
    assert mix.lam == 0.01**2, "class object lambda is not set."
    assert mix.inform_prior == False, "class object inform_prior is not set."

    # Train the model
    #
    # The GitHub action runners can have as few as two processors.  When tests
    # run on those with Open MPI with more MPI processes than processors, it
    # exits due to oversubscription.  The value of tc is set to get all Open
    # MPI-based actions running.
    fit = mix.train(
        x_train=x_train,
        y_train=y_train,
        f_train=f_train,
        ndpost=10000,
        nadapt=2000,
        nskip=2000,
        adaptevery=500,
        minnumbot=4,
        tc = 2)

    # Check the mcmc objects
    assert mix.ndpost == 10000, "class object ndpost is not set."
    assert mix.nadapt == 2000, "class object nadapt is not set."
    assert mix.adaptevery == 500, "class object adaptevery is not set."
    assert mix.nskip == 2000, "class object nskip is not set."
    assert mix.minnumbot == 4, "class object minnumbot is not set."


# Test the mean predictions
def test_predict():
    # Get test data
    n_test = 30
    x1_test = np.outer(np.linspace(-3, 3, n_test), np.ones(n_test))
    x2_test = x1_test.copy().transpose()
    x_test = np.array([x1_test.reshape(x1_test.size,),
                      x2_test.reshape(x1_test.size,)]).transpose()
    f_test = np.concatenate([f1.evaluate(x_test)[0],f2.evaluate(x_test)[0]], axis = 1)

    # Read in test results
    pmean_test = np.loadtxt(_TEST_DATA.joinpath('2d_pmean.txt'))
    eps = 0.10

    # Get predictions
    res = mix.predict(x_test=x_test,f_test = f_test, ci=0.95)

    # Test the values
    perr = np.mean(np.abs(res["pred"]["mean"] - pmean_test))
    assert perr < eps, "Inaccurate predictions."


# Test posterior of the weights
def test_predict_wts():
    # Get weights
    n_test = 30
    x1_test = np.outer(np.linspace(-3, 3, n_test), np.ones(n_test))
    x2_test = x1_test.copy().transpose()
    x_test = np.array([x1_test.reshape(x1_test.size,),
                      x2_test.reshape(x1_test.size,)]).transpose()

    res = mix.predict_weights(x_test=x_test, ci=0.95)

    # Read in test results
    wteps = 0.05
    wmean_test = np.loadtxt(_TEST_DATA.joinpath('2d_wmean.txt'))

    # Test the values
    werr = np.mean(np.abs(res["wts"]["mean"] - wmean_test))
    assert werr < wteps, "Inaccurate weights."


# Test sigma
def test_sigma():
    sig_eps = 0.05
    n_test = 5
    x1_test = np.outer(np.linspace(-3, 3, n_test), np.ones(n_test))
    x2_test = x1_test.copy().transpose()
    x_test = np.array([x1_test.reshape(x1_test.size,),
                      x2_test.reshape(x1_test.size,)]).transpose()
    f_test = np.concatenate([f1.evaluate(x_test)[0],f2.evaluate(x_test)[0]],axis = 1)

    res = mix.predict(x_test=x_test, f_test = f_test, ci=0.95)
    assert np.abs((res["sigma"]["mean"][0] - 0.1)) < sig_eps, "Inaccurate sigma calculation."


# ---------------------------------------------
# Initiatilize model set
# ---------------------------------------------
# Define the model set
f1 = sin_cos_exp(7, 10, np.pi, np.pi)
f2 = sin_cos_exp(13, 6, -np.pi, -np.pi)

mix = Openbtmix()
