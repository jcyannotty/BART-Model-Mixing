import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openbt import OPENBT

#import subprocess
#import shutil
import importlib
import openbt
importlib.reload(openbt)
from openbt import OPENBT

# Test Branin function, rescaled
def braninsc(xx):
    x1 = xx[0] 
    x2 = xx[1]
    
    x1bar = 15 * x1 - 5
    x2bar = 15 * x2
    
    term1 = x2bar - 5.1*x1bar**2/(4*math.pi**2) + 5*x1bar/math.pi - 6
    term2 = (10 - 10/(8*math.pi)) * math.cos(x1bar)
    
    y = (term1**2 + term2 - 44.81) / 51.95
    return(y)

# Simulate branin data for testing
np.random.seed(99)
n = 500
p = 2
x = np.random.uniform(size=n*p).reshape(n,p)
y = np.zeros(n)
for i in range(n):
    y[i] = braninsc(x[i,]) + np.random.normal(0,0.5,1) 


#------------------------------------------------
# Testing BART
#------------------------------------------------
m = OPENBT(model = "bart", tc = 4, modelname = "branin", ntree = 50)
fit = m.fit(x, y)

fit['fpath']
fit['pbdh']
fit['pbh']
fit['ntreeh']
fit['minnumboth']

# Calculate in-sample predictions
fitp = m.predict(x, tc = 4)

# Plot observed vs. predicted
fig = plt.figure(figsize=(16,9)); 
ax = fig.add_subplot(111)
ax.plot(y, fitp['mmean'], 'ro')
ax.set_xlabel("Observed"); ax.set_ylabel("Fitted")
ax.axline([0, 0], [1, 1])
plt.show()

# Print predicted means
fitp['mmean']
fitp['smean']


#------------------------------------------------
# Testing MixBART
#------------------------------------------------
# Polynomial function
def fp(x,a = 0,b = 0,c = 1,p = 1):
    if isinstance(x, list):
        x = np.array(x)
    f = c*(x-a)**p + b
    return f

n_train = 15
n_test = 100
s = 0.5

x_train = np.concatenate([np.array([0.01,0.1,0.25]),np.linspace(0.45,1.0, n_train-3)])
x_test = np.linspace(0.01, 1.0, n_test)

np.random.seed(1234567)
y_train = fp(x_train, 0.5, 0, 8, 2) + np.random.normal(0,s,n_train)
f0_test = fp(x_test, 0.5, 0, 8, 2)

y_train = np.array([1.95997594,1.62345280,0.68266756,-0.31770023,-0.00212874,0.10024547,-0.36453710,
            0.40737596,0.09014892,0.46057129, 0.99549935,1.36539788,1.18544634,1.49535754,1.86467596])

f1_train = fp(x_train, 0,-2,4,1).reshape(n_train,1)
f1_test = fp(x_test, 0,-2,4,1).reshape(n_test,1)
f2_train = fp(x_train, 0,2,-4,1).reshape(n_train,1)
f2_test = fp(x_test, 0,2,-4,1).reshape(n_test,1)

f_train = np.concatenate([f1_train, f2_train], axis = 1)
f_test = np.concatenate([f1_test, f2_test], axis = 1)

mix = OPENBT(model = "mixbart", tc = 4, modelname = "parabola", ntree = 10, k = 1, ndpost = 10000, nskip = 2000, nadapt = 5000, 
                adaptevery = 500, overallsd = 0.556, minnumbot = 1)
fitx = mix.fit(x_train, y_train, f_train)
fitxp = mix.predict(X = x_test, F = f_test,tc = 4)
fitxw = mix.mixingwts(X = x_test)

# Plot function overlayed with predicted
fig = plt.figure(figsize=(16,9)); 
ax = fig.add_subplot()
ax.plot(x_test, fitxp['mmean'], 'green')
ax.plot(x_test, f0_test, 'black')
ax.plot(x_test, f_test[:,0], 'r')
ax.plot(x_test, f_test[:,1], 'b')
ax.scatter(x_train, y_train)
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.show()


# Plot weight functions with 95% intervals
fig = plt.figure(figsize=(16,9)); 
ax = fig.add_subplot()
ax.plot(x_test, fitxw['wmean'][:,0], 'red')
ax.plot(x_test, fitxw['wmean'][:,1], 'blue')
ax.set_xlabel("x"); ax.set_ylabel("w(x)")
plt.show()

fitxp['mmean']
fitxp['smean']
fitxw['wmean']


#--------------------------------------------------
# For Taweret 
from pandas.core.frame import DataFrame as Dfclass
Dfclass([1,2])
pd.DataFrame([1,2])
