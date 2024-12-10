#example_model_mixing.R
setwd("/home/johnyannotty/Documents/Open BT Project SRC")

# Load the R wrapper functions to the OpenBT library.
source("/home/johnyannotty/Documents/Open BT Project SRC/openbt.R")
source("/home/johnyannotty/Documents/Model Mixing BART/Model Mixing R Code/Model Mixing BART Testing Priors.R")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Construct the required functions
#True function
fg = function(g){
  x = 1/(32*g^2)
  K = besselK(x = x, nu = 0.25)
  b = exp(x)/(2*sqrt(2)*g)*K
  return(b)
}

#Construct the small-g expansion
fsg = function(g, ns){
  #Get the term number
  k = 0:ns
  
  #Get the coefficients -- only even coefficients are non-zero
  sk = ifelse(k %% 2 == 0,sqrt(2)*gamma(k + 1/2)*(-4)^(k/2)/factorial(k/2), 0)
  
  #Get the expansion
  f = sum(sk*g^k)
  return(f)
}

#Construct the large-g expansion
flg = function(g, nl){
  #Get the term number
  k = 0:nl
  
  #Get the coefficients
  lk = gamma(k/2 + 0.25)*(-0.5)^k/(2*factorial(k))
  
  #Get the expansion
  f = sum(lk*g^(-k))/sqrt(g)
  return(f)
}

#Construct the small-g discrepancy
dsg = function(g,ns){
  #Get the term number
  k = 0:ns
  
  #Get the coefficients -- only even coefficients are non-zero
  sk = ifelse(k %% 2 == 0,sqrt(2)*gamma(k + 1/2)*(-4)^(k/2)/factorial(k/2), 0)
  
  #Estimate cbar
  cbar = sqrt(mean(sk^2))
  
  #get the variance estimate 
  if((ns/2)%%2 == 0){
    v = (cbar^2)*(factorial(ns/2 + 2)^2)*g^(ns + 4)
  }else{
    v = (cbar^2)*(factorial(ns/2 + 1)^2)*g^(ns + 2)
  }
  s = sqrt(v)
  return(s)
}

#Construct the large-g discrepancy
dlg = function(g,nl){
  #Get the term number
  k = 0:nl
  
  #Get the coefficients
  lk = gamma(k/2 + 0.25)*(-0.5)^k/(2*factorial(k))
  
  #Estimate cbar
  if(nl < 2){
    print("Warning, nl < 2, dbar is not estimated as intended")
    dbar = 1
  }else{
    #Estimate cbar using coefs of order 2 through nl
    dbar = sqrt(mean(lk[-c(1,2)]^2))  
  }
  
  #Get standard deviation
  v = (dbar^2)*(1/(factorial(nl+1)^2))*(1/g^(2*nl + 3))  
  s = sqrt(v)
  return(s)
}

#Test out the functions and approximations
fg(0.1)
fsg(0.1, 4)
fg(0.3)
flg(0.3, 5)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Graph the expansions and the true function.
#Create a grid of g's
g_grid = seq(0.01, 0.5, length = 300)

#Get a simple plot using only one large and one small g expansion
plot_exp = function(g_grid, ns, nl){
  #Get labels
  fs_name = paste0('Fs(',ns,')')
  fl_name = paste0('Fl(',nl,')')
  #Evaluate the functions 
  f = fg(g_grid)
  fs = sapply(g_grid, fsg, ns = ns)
  fl = sapply(g_grid, flg, nl = nl)
  plot(g_grid, f, main = 'F(g) and Exansions vs. g', xlab = 'g', ylab = 'F(g)', 
       ylim = c(1,4), type = 'l', panel.first = {grid(col = 'lightgrey')})
  lines(g_grid, fs, col = 'red', lty = 'dashed')
  lines(g_grid, fl, col = 'blue', lty = 'dashed')
  legend('bottomright', legend = c('F(g)', fs_name, fl_name), cex = 0.75, 
         lty = c(1, 2, 2), col = c('black', 'red', 'blue'))
}

plot_exp(g_grid, ns = 2, nl = 2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Generate training data
#Set parameters
n_train = 20
n_test = 300
s = 0.005 #s = 0.03

set.seed(321)
x_train = seq(0.03, 0.5, length = n_train)
#x_train = runif(n_train,0.03,0.5)
#x_train = x_train[order(x_train)]
y_train = fg(x_train) + rnorm(n_train, 0, s)

#Set a grid of test points
x_test = seq(0.03, 0.5, length = n_test)
fg_test = fg(x_test)
fs_test = sapply(x_test, fsg, 2)
fl_test = sapply(x_test, flg, 4)

#Plot the training data
plot(x_train, y_train, pch = 16, cex = 0.8, main = 'Training data')
lines(g_grid, fg(g_grid))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Define the Model set -- a matrix where each column is an the output evaluated at the train/test pts
#small_g = c(2,4,6)
#large_g = c(3,6,9)
small_g = c(2,4,6) #Which small-g expansions to use
large_g = NULL #Which large-g expansions to use
g_exp = c(small_g, large_g) #Mix both small and large g
K = length(g_exp) 
Ks = length(small_g) #Number of small-g models
Kl = length(large_g) #Number of large-g models

#Define matrices to store the function values
f_train = matrix(0, nrow = n_train, ncol = K)
f_test = matrix(0, nrow = n_test, ncol = K)

#Define discrepancy information
f_train_dmean = matrix(0, nrow = n_train, ncol = K)
f_train_dsd = matrix(1, nrow = n_train, ncol = K)
f_test_dmean = matrix(0, nrow = n_test, ncol = K)
f_test_dsd = matrix(1, nrow = n_test, ncol = K)

#Computation
for(i in 1:K){
  #Get the small g expansion output
  if(i <= length(small_g)){
    f_train[,i] = sapply(x_train, fsg, ns = g_exp[i])
    f_test[,i] = sapply(x_test, fsg, ns = g_exp[i])
    
    f_train_dsd[,i] = sapply(x_train, dsg, ns = g_exp[i])
    f_test_dsd[,i] = sapply(x_test, dsg, ns = g_exp[i])
  }else{
    #Get the large g expansion output  
    f_train[,i] = sapply(x_train, flg, nl = g_exp[i])
    f_test[,i] = sapply(x_test, flg, nl = g_exp[i])
    
    f_train_dsd[,i] = sapply(x_train, dlg, nl = g_exp[i])
    f_test_dsd[,i] = sapply(x_test, dlg, nl = g_exp[i])
  }
}

#Cast x_train and x_test as matrices
x_train = as.matrix(x_train, ncol = 1)
x_test = as.matrix(x_test, ncol = 1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Model Mixing with OpenBT 
fit=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),ntree = 10,ntreeh=1,numcut=300,tc=4,model="mixbart",modelname="physics_model",
           ndpost = 10000, nskip = 1000, nadapt = 5000, adaptevery = 500, printevery = 500,
           power = 1.0, minnumbot = 2, overallsd = 0.01, k = 3)

#Get mixed mean function
fitp=predict.openbt(fit,x.test = x_test, f.test = f_test,tc=4)

plot(x_test, fg(x_test), pch = 16, cex = 0.8, main = 'Fits', type = 'l', ylim = c(1.8,2.8))
points(x_train, y_train, pch = 3)
lines(x_test, f_test[,1], col = 'red', lty = 2)
lines(x_test, f_test[,2], col = 'blue', lty = 2)
lines(x_test, fitp$mmean, col = 'green4', lwd = 2)
lines(x_test, fitp$m.lower, col = 'orange', lwd = 2)
lines(x_test, fitp$m.upper, col = 'orange', lwd = 2)

#Get the model weights
fitw = openbt.mixingwts(fit, x.test = x_test, numwts = K, tc = 4)

#Plot model weights
plot(x_test, fitw$wmean[,1], lwd = 2, col = 'red', type = 'l', ylim = c(-0.15,1.05))
lines(x_test, fitw$wmean[,2], col = 'blue', lwd = 2)
lines(x_test, fitw$w.upper[,1], col = 'red', lty = 'dashed')
lines(x_test, fitw$w.lower[,1], col = 'red', lty = 'dashed')
lines(x_test, fitw$w.upper[,2], col = 'blue', lty = 'dashed')
lines(x_test, fitw$w.lower[,2], col = 'blue', lty = 'dashed')
#lines(x_test, fitw$wmean[,3], col = 'green', pch = 16)
#lines(x_test, fitw$w.upper[,3], col = 'green', lty = 'dashed')
#lines(x_test, fitw$w.lower[,3], col = 'green', lty = 'dashed')

#Trace Plot of the weights
plot(fitw$wdraws[[1]][,55], type = 'l')
plot(fitw$wdraws[[1]][,150], type = 'l')
plot(fitw$wdraws[[1]][,185], type = 'l')
plot(fitw$wdraws[[2]][,20], type = 'l')
plot(fitw$wdraws[[2]][,150], type = 'l')
plot(fitw$wdraws[[2]][,275], type = 'l')
wts_trace = list(wt1 = fitw$wdraws[[1]][,c(25,150,275)], wt2 = fitw$wdraws[[2]][,c(25,150,275)]) 

#Trace plots of thetas
tnp10 = read.table(paste0(fit$folder,"/physics_model.tnp1draws0"))
tnp13 = read.table(paste0(fit$folder,"/physics_model.tnp1draws3"))
tnp20 = read.table(paste0(fit$folder,"/physics_model.tnp2draws0"))
tnp23 = read.table(paste0(fit$folder,"/physics_model.tnp2draws3"))
tnp_list = list(tnp10, tnp13, tnp20, tnp23)

par(mfrow = c(2,3))
mu_trace_ind = c(1,7,15)
#mu_trace_ind = 1:15
for(i in 1:4){
  for(j in mu_trace_ind){
    temp = tnp_list[[i]] 
    plot(temp[,j], type ='l')
  }  
}
mu1_matrix = cbind(tnp_list[[1]][,mu_trace_ind],tnp_list[[3]][,mu_trace_ind])
mu2_matrix = cbind(tnp_list[[2]][,mu_trace_ind],tnp_list[[4]][,mu_trace_ind])
colnames(mu1_matrix) = colnames(mu2_matrix) = paste0(rep(c('LowX.', 'HighX.'),each = 3), colnames(mu1_matrix))
tnp_trace = list(mu1_matrix = mu1_matrix, mu2_matrix = mu2_matrix)

#Save results
fitp_out = list(mmean = fitp$mmean,m.lower = fitp$m.lower, m.upper = fitp$m.upper)
fitw_out = list(wmean = fitw$wmean,w.lower = fitw$w.lower, w.upper = fitw$w.upper)
fittpn_out =  tnp_trace
#saveRDS(fitp_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Stationary Prior April 2022/Stationary Prior 04-26/sg246_1p_0426.rds")
#saveRDS(fitw_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Stationary Prior April 2022/Stationary Prior 04-26/sg246_1w_0426.rds")
#saveRDS(fittpn_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Stationary BART priors 03-01/sg2sg2_1tnp.rds")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Model Mixing with OpenBT - With Discrepancy
f_train[,2] = rev(fg(x_train) - f_train[,1]) + fg(x_train)
f_train_dsd[,2] = rev(f_train_dsd[,1])
f_test[,2] = rev(fg(x_test) -f_test[,1]) + fg(x_test)
f_test_dsd[,2] = rev(f_test_dsd[,1])
maxf = max(f_train)
#f_test[,2] = ifelse(f_test[,2]>20,20,f_test[,2])

fit=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),ntree = 10, ntreeh=1,numcut=300,tc=4,model="mixbart",modelname="physics_model",
           ndpost = 10000, nskip = 2000, nadapt = 5000, adaptevery = 500, printevery = 500,
           power = 1.0, minnumbot = 1, overallsd = sd(y_train)/sqrt(300), k = 3, summarystats = FALSE,
           f.discrep.mean =  f_train_dmean, f.discrep.sd = f_train_dsd)

#Get mixed mean function
fitp=predict.openbt(fit,x.test = x_test, f.test = f_test, f.discrep.mean =  f_test_dmean, f.discrep.sd = f_test_dsd ,tc=4, q.lower = 0.025, q.upper = 0.975)

plot(x_test, fg(x_test), pch = 16, cex = 0.8, main = 'Fits', type = 'l', ylim = c(1.8,2.8))
points(x_train, y_train, pch = 3)
lines(x_test, f_test[,1], col = 'red', lty = 2)
lines(x_test, f_test[,2], col = 'blue', lty = 2)
lines(x_test, fitp$mmean, col = 'green4', lwd = 2)
lines(x_test, fitp$m.lower, col = 'orange', lwd = 2)
lines(x_test, fitp$m.upper, col = 'orange', lwd = 2)

#Get the model weights
fitw = openbt.mixingwts(fit, x.test = x_test, numwts = K, tc = 4)

#Plot model weights
plot(x_test, fitw$wmean[,1], pch = 16, col = 'red', type = 'l', ylim = c(-0.05,1.05), lwd = 2, 
     panel.first = {grid(col = 'lightgrey')})
lines(x_test, fitw$wmean[,2], col = 'blue', pch = 16, lwd = 2)
lines(x_test, fitw$w.upper[,1], col = 'red', lty = 'dashed', lwd = 1)
lines(x_test, fitw$w.lower[,1], col = 'red', lty = 'dashed', lwd = 1)
lines(x_test, fitw$w.upper[,2], col = 'blue', lty = 'dashed', lwd = 1)
lines(x_test, fitw$w.lower[,2], col = 'blue', lty = 'dashed', lwd = 1)
abline(h = 1, col = 'grey', lty = 'dashed')
abline(h = 0, col = 'grey', lty = 'dashed')

#Get prior weights pointwise
w1 = (1/f_train_dsd[,1]^2)/(1/f_train_dsd[,1]^2 + 1/f_train_dsd[,2]^2)
points(x_train, w1, pch = 2, col = 'red')
points(x_train, 1-w1, pch = 2, col = 'blue')

#Plot the prior fit
bhat_matrix = cbind(w1,1-w1)
plot_prior_pw_fit(bhat_matrix, f_train, x_train, f_test, x_test, y_lim = c(1.8,2.8))
plot_prior_pw_wts(bhat_matrix, x_train, y_lim = c(-0.05, 1.05))

wts_trace = list(wt1 = fitw$wdraws[[1]][,c(25,150,275)], wt2 = fitw$wdraws[[2]][,c(25,150,275)]) 
plot(fitw$wdraws[[1]][,119],type='l')

#Look at variances
apply(fitw$wdraws[[2]],2,var)
cbind(x_test[20:50],apply(fitw$wdraws[[2]],2,var)[20:50], f_test[20:50,2])
v = 20
x_test[v]
sqrt(t(f_test[v,])%*%cov(data.frame(fitw$wdraws[[1]][,v], fitw$wdraws[[2]][,v]))%*%(f_test[v,]))

#Trace plots of thetas
tnp10 = read.table(paste0(fit$folder,"/physics_model.tnp1draws0"))
tnp13 = read.table(paste0(fit$folder,"/physics_model.tnp1draws3"))
tnp20 = read.table(paste0(fit$folder,"/physics_model.tnp2draws0"))
tnp23 = read.table(paste0(fit$folder,"/physics_model.tnp2draws3"))
tnp_list = list(tnp10, tnp13, tnp20, tnp23)

par(mfrow = c(2,3))
mu_trace_ind = c(1,7,15)
#mu_trace_ind = 1:15
for(i in 1:4){
  for(j in mu_trace_ind){
    temp = tnp_list[[i]] 
    plot(temp[,j], type ='l')
  }  
}
mu1_matrix = cbind(tnp_list[[1]][,mu_trace_ind],tnp_list[[3]][,mu_trace_ind])
mu2_matrix = cbind(tnp_list[[2]][,mu_trace_ind],tnp_list[[4]][,mu_trace_ind])
colnames(mu1_matrix) = colnames(mu2_matrix) = paste0(rep(c('LowX.', 'HighX.'),each = 3), colnames(mu1_matrix))
tnp_trace = list(mu1_matrix = mu1_matrix, mu2_matrix = mu2_matrix)

#Save the results 
#openbt.save(fit,"/home/johnyannotty/Documents/Model Mixing BART/smallg2largeg4")
fitp_out = list(mmean = fitp$mmean,m.lower = fitp$m.lower, m.upper = fitp$m.upper)
fitw_out = list(wmean = fitw$wmean,w.lower = fitw$w.lower, w.upper = fitw$w.upper, wts_trace = wts_trace)
fittpn_out =  tnp_trace
#saveRDS(fitp_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Non-Stationary Priors 03-28/sg4lg4_4p.rds")
#saveRDS(fitw_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Non-Stationary Priors 03-28/sg4lg4_4w.rds")
#saveRDS(fittpn_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Stationary priors 03-01/sg2lg4_1tnp.rds")

fitp_out = list(mmean = fitp$mmean,m.lower = fitp$m.lower, m.upper = fitp$m.upper)
fitw_out = list(wmean = fitw$wmean,w.lower = fitw$w.lower, w.upper = fitw$w.upper)
#saveRDS(fitp_out, "/home/johnyannotty/Documents/Model Mixing BART/BAND Presentation Data/JAM 04-07/sg2lg4_10tree_p.rds")
#saveRDS(fitw_out, "/home/johnyannotty/Documents/Model Mixing BART/BAND Presentation Data/JAM 04-07/sg2lg4_10tree_w.rds")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Model Mixing with OpenBT -- Read in priors
#prior_mean = c(0.05,0.001) #rep(0,2)
#prior_sd = c(0.045,0.02) #rep((max(y_train)-min(y_train))/(2*1.5*sqrt(20)), 2)
#prior_info = cbind(
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~prior_mean, prior_sd)
m = 20
k = 2.25
#prior_out = prior1(f_train, y_train, k,m)
prior_out = prior2(f_train, y_train, k,m)
prior_info = cbind(prior_out$beta_list, prior_out$tau_list)

fit=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),ntree = m, ntreeh=1,numcut=300,tc=4,model="mixbart",modelname="physics_model",
           ndpost = 10, nskip = 1000, nadapt = 5000, adaptevery = 500, printevery = 500,
           power = 1.0, minnumbot = 5, overallsd = sd(y_train)/sqrt(16), k = k, f.prior.info = prior_info)

#Get mixed mean function
fitp=predict.openbt(fit,x.test = x_test, f.test = f_test,tc=4)

plot(g_grid, fg(g_grid), pch = 16, cex = 0.8, main = 'Fits', type = 'l', ylim = c(1.8,2.8))
points(x_train, y_train, pch = 3)
lines(g_grid, sapply(g_grid,fsg, small_g), col = 'red', lty = 2)
lines(g_grid, sapply(g_grid,flg, large_g), col = 'blue', lty = 2)
points(x_test, fitp$mmean, col = 'green4')
points(x_test, fitp$m.lower, col = 'orange')
points(x_test, fitp$m.upper, col = 'orange')

#Get the model weights
fitw = openbt.mixingwts(fit, x.test = x_test, numwts = 2, tc = 4)

#Plot model weights
plot(x_test, fitw$wmean[,1], pch = 16, col = 'red', type = 'o', ylim = c(-1,1))
points(x_test, fitw$wmean[,2], col = 'blue', pch = 16)
lines(x_test, fitw$w.upper[,1], col = 'red', lty = 'dashed')
lines(x_test, fitw$w.lower[,1], col = 'red', lty = 'dashed')
lines(x_test, fitw$w.upper[,2], col = 'blue', lty = 'dashed')
lines(x_test, fitw$w.lower[,2], col = 'blue', lty = 'dashed')

#Trace Plot of the weights
plot(fitw$wdraws[[1]][,25], type = 'l')
plot(fitw$wdraws[[1]][,185], type = 'l')
plot(fitw$wdraws[[2]][,25], type = 'l')
plot(fitw$wdraws[[2]][,185], type = 'l')

wts_trace = list(wt1 = fitw$wdraws[[1]][,c(25,150,275)], wt2 = fitw$wdraws[[2]][,c(25,150,275)]) 

#Trace plots of thetas
tnp10 = read.table(paste0(fit$folder,"/physics_model.tnp1draws0"))
tnp13 = read.table(paste0(fit$folder,"/physics_model.tnp1draws1"))
tnp20 = read.table(paste0(fit$folder,"/physics_model.tnp2draws0"))
tnp23 = read.table(paste0(fit$folder,"/physics_model.tnp2draws1"))
tnp_list = list(tnp10, tnp13, tnp20, tnp23)

mu_trace_ind = c(1,7,15)
#mu_trace_ind = 1:15
for(i in 1:4){
  for(j in mu_trace_ind){
    temp = tnp_list[[i]] 
    plot(temp[,j], type ='l')
  }  
}

mu1_matrix = cbind(tnp_list[[1]][,mu_trace_ind],tnp_list[[3]][,mu_trace_ind])
mu2_matrix = cbind(tnp_list[[2]][,mu_trace_ind],tnp_list[[4]][,mu_trace_ind])
colnames(mu1_matrix) = colnames(mu2_matrix) = paste0(rep(c('LowX.', 'HighX.'),each = 3), colnames(mu1_matrix))
tnp_trace = list(mu1_matrix = mu1_matrix, mu2_matrix = mu2_matrix)

#Save results
fitp_out = list(mmean = fitp$mmean,m.lower = fitp$m.lower, m.upper = fitp$m.upper)
fitw_out = list(wmean = fitw$wmean,w.lower = fitw$w.lower, w.upper = fitw$w.upper, wts_trace = wts_trace)
fittpn_out =  tnp_trace
#saveRDS(fitp_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Stationary BART priors 03-01/sg2sg2_4p.rds")
#saveRDS(fitw_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Stationary BART priors 03-01/sg2sg2_4w.rds")
#saveRDS(fittpn_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Stationary BART priors 03-01/sg2sg2_4tnp.rds")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Check calculations
inv_prior_var = diag(c(30,60))
beta0 = c(0,0)
data_rng = c(1,5)
sig2 = 0.3
solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)%*%(t(f_train[data_rng,])%*%y_train[data_rng]/sig2 + inv_prior_var%*%beta0)
solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)

sqrt(solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)[1,1])
sqrt(solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)[2,2])

b = solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)%*%(t(f_train[data_rng,])%*%y_train[data_rng]/sig2 + inv_prior_var%*%beta0)
f_train[data_rng,]%*%b

bhat = solve(t(f_train[data_rng,])%*%f_train[data_rng,])%*%t(f_train[data_rng,])%*%y_train[data_rng]
f_train[data_rng,]%*%bhat

#G prior
g = 10
q = g/(g+1)
bhat = solve(t(f_train[data_rng,])%*%f_train[data_rng,])%*%t(f_train[data_rng,])%*%y_train[data_rng]
q*bhat

#use discrepancies for the prior
data_rng = c(8:13)
#tau2 = (0.6/2)^2
tau2 = 1/(2^2)
inv_prior_var = diag(c(sum(f_train_dsd[data_rng,1]^2),sum(f_train_dsd[data_rng,2]^2)))/tau2
beta0 = c(0.5,0.5)
sig2 = 0.56
solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)%*%(t(f_train[data_rng,])%*%y_train[data_rng]/sig2 + inv_prior_var%*%beta0)
solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)

sqrt(solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)[1,1])
sqrt(solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)[2,2])

b = solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)%*%(t(f_train[data_rng,])%*%y_train[data_rng]/sig2 + inv_prior_var%*%beta0)
f_train[data_rng,]%*%b
bhat = solve(t(f_train[data_rng,])%*%f_train[data_rng,])%*%t(f_train[data_rng,])%*%y_train[data_rng]
f_train[data_rng,]%*%bhat

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Priors
data_rng = c(2:3)
pr = 1
bartp = bart_prior(y_train[data_rng], c=2,m=1,K=2)
nsp1 = ns_prior1(dsd.train = f_train_dsd[data_rng,], 2,1)
nsp2 = ns_prior2(dsd.train = f_train_dsd[data_rng,], 5,1)

sig2 = 0.01
if(pr == 1){
  inv_prior_var = diag(1/diag(nsp1$tau_matrix),2)^2
  beta0 = nsp1$beta
}else{
  inv_prior_var = diag(1/diag(nsp2$tau_matrix),2)^2
  beta0 = nsp2$beta  
}

b = solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)%*%(t(f_train[data_rng,])%*%y_train[data_rng]/sig2 + inv_prior_var%*%beta0)
f_train[data_rng,]%*%b

bhat = solve(t(f_train[data_rng,])%*%f_train[data_rng,])%*%t(f_train[data_rng,])%*%y_train[data_rng]
f_train[data_rng,]%*%bhat

sqrt(solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)[1,1])
sqrt(solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)[2,2])
