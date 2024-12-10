import pandas as pd
import numpy as np
import math
import scipy
import gsum as gm
import matplotlib.pyplot as plt

from itertools import chain
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

#------------------------------------------------------
# Define Small-g expansion terms -- using coefs and Q(x)
def fsg_terms(x,k):
    #Get the coefficients -- only even coefficients are non-zero
    # Get q(x) and yref(x)
    if k % 2 == 0:
        sk = ((-2)**(k/2))*scipy.special.gamma(k + 1/2)/math.factorial(k/2)
        #sk = np.sqrt(2)*scipy.special.gamma(k + 1/2)*(-4)**(k/2)/math.factorial(k/2)
        qx = np.sqrt(2)*x**0.5
        yref = np.sqrt(2)
    else:
        sk = 0
        qx = 0
        yref = 0
    # Get the polynomial term
    cx = sk*x**(k/2)
    tx = cx*yref*qx**k
    out = {'tx':tx,'cx':cx,'qx':qx,'yref':yref}
    return out

# Define Large-g expansion terms -- using coefs and Q(x)
def flg_terms(x,k):
    #Get the coefficients
    lk = scipy.special.gamma(k/2 + 0.25)*(-1)**(k)/(2*math.factorial(k))
  
    #Get the expansion coef, q, and term
    yref = 1/np.sqrt(x)
    qx = 0.5
    cx = lk*x**(-k)
    tx = yref*cx*qx**k

    out = {'tx':tx, 'cx':cx, 'qx':qx, 'yref':yref}
    return out


# Get polynomial terms for the model at the specified inputs
def get_terms(x_list,k, model = 'sg'):
    term_list = []
    coef_list = []
    qx_list = []
    yref_list = []
    for x in x_list:
        if model == 'sg':
            res = fsg_terms(x,k)
        elif model == 'lg':
            res = flg_terms(x,k)
        else:
            raise ValueError('Wrong model specified. Valid arguments are either sg or lg.') 
        tx = res['tx']
        cx = res['cx'] 
        qx = res['qx']
        yref = res['yref']
        term_list.append(tx)
        coef_list.append(cx)
        qx_list.append(qx)
        yref_list.append(yref)
      
    out = {'term_list':term_list, 'coef_list':coef_list, 'qx_list':qx_list, 'yref_list':yref_list}
    return out

# Define the expansions
def get_exp(x_list,k, model = 'sg'):
    cols = []
    for i in list(range(0,k+1)):
        cols.append('order'+str(i))
    coef_df = pd.DataFrame(columns = cols)
    term_df = pd.DataFrame(columns = cols)
    ypred_df = pd.DataFrame(columns = cols)
    qx_df = pd.DataFrame(columns = ['qx'])
    yref_df = pd.DataFrame(columns = ['yref'])
    for i in range(0,k+1):
        res = get_terms(x_list, i, model)
        term_df.iloc[:,i] = res['term_list']
        coef_df.iloc[:,i] = res['coef_list']
        if i == k:
            qx_df.iloc[:,0] = res['qx_list']
            yref_df.iloc[:,0] = res['yref_list']
        if i >= 1:
            ypred_df.iloc[:,i] = term_df.iloc[:,i] + ypred_df.iloc[:,i-1]
        else:
            ypred_df.iloc[:,i] = term_df.iloc[:,i]
    out = {'coef_df':coef_df, 'term_df':term_df, 'ypred_df':ypred_df, 'qx_df':qx_df,'yref_df':yref_df}
    return out 

# Checker
fsg_terms(0.25, 6)
flg_terms(0.03, 4)

# Set training points for model runs
#x1_train = np.linspace(0.05, 0.5, 5)
#x2_train = np.linspace(0.05, 0.5, 5)

x1_train = np.array([0.05, 0.15, 0.25, 0.35, 0.45])
x2_train = np.array([0.1, 0.2,0.30,0.40])

x1_test = np.linspace(0.01, 0.48, 20)
x2_test = np.linspace(0.03, 0.5, 20)

#x1_test = np.linspace(0.03, 0.5, 300)
#x2_test = np.linspace(0.03, 0.5, 300)

x1_data = np.concatenate((x1_train,x1_test), axis = 0)
x2_data = np.concatenate((x2_train,x2_test), axis = 0)

#Get coefficients from both models
ns = 4
c1_cols = []
c1_list = []
for i in range(ns+1): 
    c1_cols.append('order' + str(i))
    c1_data = get_terms(x1_train, i,'sg')['coef_list']
    c1_list.append(c1_data)
c1_array = np.array(c1_list).transpose()

nl = 4
c2_cols = []
c2_list = []
for i in range(nl+1): 
    c2_cols.append('order' + str(i))
    c2_data = get_terms(x2_train, i,'lg')['coef_list']
    c2_list.append(c2_data)
c2_array = np.array(c2_list).transpose()

# Convert the x's to required format
x1_train = x1_train[:,None]
x2_train = x2_train[:,None]

x1_test = x1_test[:,None]
x2_test = x2_test[:,None]

x1_data = x1_data[:,None]
x2_data = x2_data[:,None]

#-------------------------
# Hyperparameters
center0 = 0
disp0 = 0
df0 = 3
scale0 = 5
kernel1 = RBF(length_scale=0.3, length_scale_bounds=(0.05, 4)) + WhiteKernel(1e-10, noise_level_bounds='fixed')
kernel2 = RBF(length_scale=0.5, length_scale_bounds=(0.05, 1)) + WhiteKernel(1e-10, noise_level_bounds='fixed')

exp1_res = get_exp(list(chain(*x1_data.tolist())), ns, 'sg')

ratio1 = np.array(list(chain(*exp1_res['qx_df'].values.tolist()))) # Reshape the array
ref1 = np.array(list(chain(*exp1_res['yref_df'].values.tolist()))) # Reshape the array

y1_df = exp1_res['ypred_df']
coeffs1 = gm.coefficients(np.array(y1_df),ratio = ratio1,ref=ref1,orders=np.arange(0,ns+1))

gp1 = gm.ConjugateGaussianProcess(kernel1, center=center0, disp=disp0, df=df0, scale=scale0, n_restarts_optimizer=10)
fit1 = gp1.fit(x1_train, coeffs1[:len(x1_train),])
fit1.predict(x1_test,return_std=True)
fit1.cbar_sq_mean_
fit1.scale_
fit1.df_

gp_trunc1 = gm.TruncationGP(kernel = fit1.kernel_, center=center0, disp=disp0, df=fit1.df_, scale=fit1.scale_, ref = ref1, ratio = ratio1)

fhat1 = y1_df.iloc[:,ns] + gp_trunc1.predict(x1_data, order = ns, return_std = False)
std1 = gp_trunc1.predict(x1_data, order = ns, return_std = True, kind = 'trunc')[1]
cov1 = gp_trunc1.predict(x1_data, order = ns, return_cov = True)[1]

#fit_trunc1 = gp_trunc1.fit(x1_data, y=np.array(exp1_res['ypred_df']), orders=np.arange(0,ns+1))
#fhat1 = fit_trunc1.predict(x1_data, order = 2, return_std = False)
#std1 = fit_trunc1.predict(x1_data, order = 2, return_std = True, kind = 'trunc')[1]
#cov1 = fit_trunc1.predict(x1_data, order = 2, return_cov = True)[1]


fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax.plot(x1_test, fhat1[len(x1_train):], zorder = 2)
ax.fill_between(x1_test[:,0], fhat1[len(x1_train):] + 2*std1[len(x1_train):], fhat1[len(x1_train):] - 2*std1[len(x1_train):], zorder=1)
plt.show()

fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax.plot(x1_test, exp1_res['coef_df'].iloc[len(x1_train):,2], zorder = 2)
ax.plot(x1_test, exp1_res['coef_df'].iloc[len(x1_train):,4], zorder = 2, color = 'red')
#ax.plot(x1_test, exp1_res['coef_df'].iloc[len(x1_train):,6], zorder = 2, color = 'green')
ax.plot(x1_test, exp1_res['coef_df'].iloc[len(x1_train):,0], zorder = 2, color = 'purple')
plt.show()


center0 = 0
disp0 = 0
df0 = 3
scale0 = 5

exp2_res = get_exp(list(chain(*x2_data.tolist())), 4, 'lg')
y2_df = exp2_res['ypred_df']

ratio2 = np.array(list(chain(*exp2_res['qx_df'].values.tolist()))) # Reshape the array
ref2 = np.array(list(chain(*exp2_res['yref_df'].values.tolist()))) # Reshape the array

coeffs2 = gm.coefficients(np.array(y2_df),ratio = ratio2,ref = ref2, orders=np.arange(0,nl+1))

gp2 = gm.ConjugateGaussianProcess(kernel2, center=center0, disp=disp0, df=df0, scale=scale0, n_restarts_optimizer=10)
fit2 = gp2.fit(x2_train, coeffs2[:len(x2_train),])
fit2.cbar_sq_mean_

gp_trunc2 = gm.TruncationGP(kernel = fit2.kernel_, center=center0, disp=disp0, df=fit2.df_, scale=fit2.scale_, ref = ref2, ratio = ratio2)
fhat2 = y2_df.iloc[:,ns] + gp_trunc2.predict(x2_data, order = nl, return_std = False)
std2 = gp_trunc2.predict(x2_data, order = nl, return_std = True, kind = 'trunc')[1]
cov2 = gp_trunc2.predict(x2_data, order = nl, return_cov = True)[1]

#fit_trunc1 = gp_trunc1.fit(x1_data, y=np.array(exp1_res['ypred_df']), orders=np.arange(0,ns+1))
#fhat1 = fit_trunc1.predict(x1_data, order = 2, return_std = False)
#std1 = fit_trunc1.predict(x1_data, order = 2, return_std = True, kind = 'trunc')[1]
#cov1 = fit_trunc1.predict(x1_data, order = 2, return_cov = True)[1]


fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax.plot(x2_test, fhat2[len(x2_train):], zorder = 2)
ax.fill_between(x2_test[:,0], fhat2[len(x2_train):] + 2*std2[len(x2_train):], fhat2[len(x2_train):] - 2*std2[len(x2_train):], zorder=1)
plt.show()

fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax.plot(x2_test, exp2_res['coef_df'].iloc[len(x2_train):,2], zorder = 2)
ax.plot(x2_test, exp2_res['coef_df'].iloc[len(x2_train):,4], zorder = 2, color = 'red')
#ax.plot(x1_test, exp1_res['coef_df'].iloc[len(x1_train):,6], zorder = 2, color = 'green')
ax.plot(x2_test, exp2_res['coef_df'].iloc[len(x2_train):,0], zorder = 2, color = 'purple')
plt.show()



#-----------------------------------------------------
nn = fhat1.shape[0]
fdata = np.concatenate([fhat1.reshape(nn,1), fhat2.reshape(nn,1)], axis = 1)
sdata = np.concatenate([std1.reshape(nn,1), std2.reshape(nn,1)], axis = 1)

np.savetxt("/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Overleaf Results/f_train.txt", fdata[4:])
np.savetxt("/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Overleaf Results/s_train.txt", sdata[4:])


#-----------------------------------------------------
# Model Mixing
n_train = 20
n_test = 100
s = 0.03

x_train = x1_data[4:]
x_grid = x1_test
#x_test = np.linspace(0.01, 1.0, n_test)

np.random.seed(1234567)
y_train = np.array([2.508488,2.481616,2.462142,2.436121,2.405677,2.374887,2.343086,2.305914,2.271658,2.232709,2.203263,2.175726,2.136858,
    2.116650, 2.068122, 2.040274, 2.017935, 1.990049, 1.964042, 1.937480])
nn = fhat1.shape[0]
f_train = np.concatenate([fhat1.reshape(nn,1), fhat2.reshape(nn,1)], axis = 1)
s_train = np.concatenate([std1.reshape(nn,1), std2.reshape(nn,1)], axis = 1)

from openbt import OPENBT
m = OPENBT(model = "mixbart", tc = 4, modelname = "eft", ntree = 10, k = 1, ndpost = 10000, nskip = 2000, nadapt = 5000, 
                adaptevery = 500, overallsd = 0.01, minnumbot = 3, nsprior = True, overallnu = 5, numcut = 300)
fit = m.fit(x_train, y_train, F= f_train[4:(n_train+4),:], S= s_train[4:(n_train+4),:])
f_grid = f_train[(n_train+4):,:]
fitp = m.predict(x_grid, F = f_grid)

fitp['mmean']

fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.plot(x_grid, fitp['mmean'], zorder = 2)
ax.plot(x_grid, fitp['mmean'], zorder = 2)
ax.plot(x_grid, fitp['m_lower'], zorder = 2)
ax.plot(x_grid, fitp['m_upper'], zorder = 2)
plt.show()














#-------------------------

fit1.kernel_
fit1.kernel_.__call__(x1_test)
fit1.kernel_.__call__(x1_train)
fit1.cov(x1_train)
fit1.cov(x1_test)
fit1.cov(x1_test, x1_train)
dir(fit1)

np.sqrt(fit1.cbar_sq_mean_)
np.sqrt(fit1.df_ * fit1.scale_**2 / (fit1.df_ + 2)) # Posterior mode estimate of cbar

k1_out = kernel1.get_params()
k1_out['k1']
kernel1.__call__(x1_test)



# Not what we want
# Get truncation errors for both models -- stop using this chunk
gp_trunc1 = gm.TruncationGP(kernel = kernel1, center=center0, disp=disp0, df=df0, scale=scale0)
gp_trunc2 = gm.TruncationGP(kernel = kernel2, center=center0, disp=disp0, df=df0, scale=scale0)
fit_trunc1 = gp_trunc1.fit(x1_train, y=c1_array, orders=np.arange(0,ns+1))
fit_trunc2 = gp_trunc2.fit(x2_train, y=c2_array, orders=np.arange(0,nl+1))

fit1.predict(x1_train, order=2, return_std=True, kind='trunc')
fit2.predict(x2_train, order=4, return_std=True, kind='trunc') 




#-----------------------------------------------------------
# Paper example
x = np.linspace(0, 1, 100)
X = x[:, None]  # make a 2D version of x to match the input data structure from SciKitLearn
n_orders = 4    # Here we examine the case where we have info on four non-trivial orders
orders = np.arange(0, n_orders)

final_order = 20  # We are going to treat the order-20 result as the final, converged answer
orders_all = np.arange(0, final_order+1)

# The true values of the hyperparameters for generating the EFT coefficients
ls = 0.2
sd = 1
center = 0
ref = 10
ratio = 0.5
nugget = 1e-10
seed = 3

# Get train data
def regular_train_test_split(x, dx_train, dx_test, offset_train=0, offset_test=0, xmin=None, xmax=None):
    train_mask = np.array([(i - offset_train) % dx_train == 0 for i in range(len(x))])
    test_mask = np.array([(i - offset_test) % dx_test == 0 for i in range(len(x))])
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    train_mask = train_mask & (x >= xmin) & (x <= xmax)
    test_mask = test_mask  & (x >= xmin) & (x <= xmax) & (~ train_mask)
    return train_mask, test_mask

x_train_mask, x_valid_mask = regular_train_test_split(
    x, dx_train=24, dx_test=6, offset_train=1, offset_test=1)

kernel = RBF(length_scale=ls, length_scale_bounds='fixed') + \
    WhiteKernel(noise_level=nugget, noise_level_bounds='fixed')
gp = gm.ConjugateGaussianProcess(kernel=kernel, center=center, df=np.inf, scale=sd, nugget=0)

# Draw coefficients and then use `partials` to create the sequence of partial sums
# that defines the order-by-order EFT predictions
# Negative sign to make y_n positive, for aesthetics only
coeffs_all = - gp.sample_y(X, n_samples=final_order+1, random_state=seed)
data_all = gm.partials(coeffs_all, ratio, ref=ref, orders=orders_all)
diffs_all = np.array([data_all[:, 0], *np.diff(data_all, axis=1).T]).T

# Get the "all-orders" curve
data_true = data_all[:, -1]

# Will only consider "known" lower orders for most of the notebook
coeffs = coeffs_all[:, :n_orders]
data = data_all[:, :n_orders]
diffs = diffs_all[:, :n_orders]

# Truncation errors
kernel_fit = RBF(length_scale=ls) + WhiteKernel(noise_level=nugget, noise_level_bounds='fixed')
gp_trunc = gm.TruncationGP(kernel=kernel_fit, ref=ref, ratio=ratio, center=center0, disp=disp0, df=df0, scale=scale0)
fit0 = gp_trunc.fit(X, y=data, orders=orders)
fit0.predict(X, order = 3)