from re import M
from types import ModuleType
import invoke # A task execution tool; unused
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
# ^ Two of these aren't used yet; B.E. is the parent class of OPENBT
import tempfile # Generate temporary directories/files
from pathlib import Path # To write filepaths
from collections import defaultdict # For setting dictionaries with default values; unused
from scipy.stats import spearmanr # For calculating the spearman coeff
import pickle # For writing to compressed (pickled) test files; unused
import numpy as np # For manipulating arrays, doing math, etc.
import subprocess # For running a process in the command line
from scipy.stats import norm # Similar to pnorm, rnorm, qnorm, etc. in R
import sys; import os # For exit() and checking the config files
import itertools # To makes labels for sobol variable pairs
import pandas as pd # Sobol has column names, so each returned array has to be a pandas df

class OPENBT(BaseEstimator):
    """Class to run openbtcli by using sklearn-like calls"""
    def __init__(self, **kwargs):
        self.ndpost = 1000 # All of these are defaults; overwriting comes later
        self.nskip = 100
        self.nadapt = 1000
        self.power = 2.0
        self.base = 0.95
        self.tc = 2
        self.pbd = 0.7
        self.pb = 0.5
        self.stepwpert = 0.1
        self.probchv = 0.1
        self.minnumbot = 5
        self.printevery = 100
        self.numcut = 100
        self.adaptevery = 100
        # Here are the extra parameters that I added, since I wanted to customize them:
        self.overallsd = None; self.overallnu = None
        self.k = None
        self.ntree = None; self.ntreeh = None
        self.truncateds = None
        self.nummodels = 2
        self.nsprior = False
        self.wtsprior = False
        # hyperthread = False # Supposed to let you run processes on all hyperthreads, not just each core
        # I added a few more if statements in _define_params() to make these go smoothly
        self.modelname = "model"
        self.summarystats = "FALSE"
        self.__dict__.update((key, value) for key, value in kwargs.items())
        # print(self.__dict__)
        self._set_model_type()


    def fit(self, X, y, F = None, S = None):
        """
        Writes out data and fits model. X and y should be numpy arrays, 
        but I added some checks to allow 1-D lists to be accepted. No promises on 
        attempting to pass in a grid of lists within a list as an array.
        """
        # Cast X to np.array if the input is a list
        if (type(X) == list):
            X = np.array(X)
            print("Completed list-to-numpy_array conversion for X. Be careful about row/column mixups!")
        
        # Cast y to np.array if the input is a list
        if (type(y) == list):
            y = np.array(y)
            print("Completed list-to-numpy_array conversion for y. Be careful about row/column mixups!")
        
        # Reshape X_train to be pxn --- keeping this to remain in sync with remainder of code
        self.X_train = np.transpose(X) 
        
        if (len(self.X_train.shape) == 1): # If shape is (n, ), change it to (n, 1):
            self.X_train = self.X_train.reshape(1, len(X)) 
            
        if (len(y.shape) == 1): # Same thing for y
            y = y.reshape(len(y), 1)

        # Set the y and fmean information    
        self.y_orig = y    
        self.fmean = np.mean(y)

        if self.model == 'mixbart':
            # Check if F is specified
            if F is None:
                raise ValueError("Please input the F matrix for model mixing.")
            # Set prior type (stationary vs. nonstationary)
            if S is None:
                self.nsprior = False
            else:
                self.nsprior = True
                # Check the dimensions of F and S
                if (not F.shape == S.shape):
                    raise ValueError("Dimensions of F and S do not match.")


            # Cast F and S to arrays (only needed for S when using nsprior)
            if isinstance(F, list):
                F = np.array(F) 
            
            if isinstance(S, list) and self.nsprior:
                S = np.array(F) 
            
            # Set the data as Class objects
            self.nummodels = F.shape[1] 
            self.F_train = F
            self.S_train = S

            # Set other default arguments
            self.wtsprior = False # T/F did user pass in list of hyperparameters for each model weight (rather than using the standard ones)

        # Overwrite any default parameters
        self._define_params()
        print("Writing config file and data")
        self._write_config_file()
        self._write_data()
        print("Running model...")
        self._run_model()
        
        # New: Return attributes to be saved as a separate fit object:
        res = {} # Missing the influence attribute from the R code (skip for now)
        self.maxx = np.ceil(np.max(self.X_train, axis=1))
        self.minx = np.floor(np.min(self.X_train, axis=1))
        keys = list(self.__dict__.keys()) # Ones we want to save into the object
        for later_key in ['p_test', 'n_test', 'q_lower', 'q_upper', 'xproot', 
            'mdraws', 'sdraws', 'mmean', 'smean', 'msd', 'ssd', 'm_5', 's_5', 
            'm_lower', 's_lower', 'm_upper', 's_upper']:
            if later_key in keys:
                keys.remove(later_key) # Taking away possible residual keys from fitp, fits, or fitv
        for key in keys:
             res[key] = self.__dict__[key]
        res['minx'] = self.minx; res['maxx'] = self.maxx
        return res


    def _set_model_type(self):
        models = {"bt": 1,
                  "binomial": 2,
                  "poisson": 3,
                  "bart": 4,
                  "hbart": 5,
                  "probit": 6,
                  "modifiedprobit": 7,
                  "merck_truncated": 8,
                  "mixbart":9
                  }
        if self.model not in models:
            raise KeyError("Not supported model type. Please specify a model.")
        self.modeltype = models[self.model]
        k_map = {1: 2, 2: 2, 3: 2, 4: 2, 5: 5, 6: 1, 7: 1, 8: 2, 9:2}
        onu_map = {1: 1, 2: 1, 3: 1, 4: 10, 5: 10, 6: -1, 7: -1, 8: 10, 9:10}
        ntree_map = {1: 1, 2: 1, 3: 1, 4: 200, 5: 200, 6: 200, 7: 200, 8: 200, 9:25}
        ntreeh_map = {1: 1, 2: 1, 3: 1, 4: 1, 5: 40, 6: 1, 7: 40, 8: 1, 9:1}
        if (self.k is None):
             print("Overwriting k to agree with the model's default")
             self.k = k_map[self.modeltype]
        if (self.overallnu is None):
             print("Overwriting overallnu to agree with the model's default")
             self.overallnu = onu_map[self.modeltype]
        if (self.ntree is None):
             print("Overwriting ntree to agree with the model's default")
             self.ntree = ntree_map[self.modeltype]
        if (self.ntreeh is None):
             print("Overwriting ntreeh to agree with the model's default")
             self.ntreeh = ntreeh_map[self.modeltype]
        # overallsd will be done in the define_params function.


    # --- the try case is checking to see if a list of length 2 is the item for each dictionary key + "h"
    def _update_h_args(self, arg):
        try:
            self.__dict__[arg + "h"] = self.__dict__[arg][1]
            self.__dict__[arg] = self.__dict__[arg][0]
        except:
            self.__dict__[arg + "h"] = self.__dict__[arg]
        # ^ Right now it seems to do the 'except' step for all args it's used with, FYI


    def _define_params(self):
        """Set up parameters for the openbtcli
        """
        if (self.modeltype in [4, 5, 8]):
           self.y_train = self.y_orig - self.fmean
           self.fmean_out = 0
           self.rgy = [np.min(self.y_train), np.max(self.y_train)]
        elif (self.modeltype in [6, 7]):
            self.fmean_out = norm.ppf(self.fmean)
            self.y_train = self.y_orig
            self.rgy = [-2, 2]
            self.uniqy = np.unique(self.y_train) # Already sorted, btw
            if(len(self.uniqy) > 2 or self.uniqy[1] != 0 or self.uniqy[2] != 1):
                 sys.exit("Invalid y.train: Probit requires dichotomous response coded 0/1") 
        elif self.modeltype in [9]:
            self.fmean_out = 0
            self.fmean = 0
            self.y_train = self.y_orig
            self.rgy = [np.min(self.y_train), np.max(self.y_train)]
        else: # Unused modeltypes for now, but still set their properties just in case
            self.y_train = self.y_orig 
            self.fmean_out = None
            self.rgy = [-2, 2] # These proprties are ambiguous for these modeltypes by the way...
            
        self.n = self.y_train.shape[0]
        self.p = self.X_train.shape[0]
        # Cutpoints
        if "xicuts" not in self.__dict__:
            self.xi = {}
            maxx = np.ceil(np.max(self.X_train, axis=1))
            minx = np.floor(np.min(self.X_train, axis=1))
            for feat in range(self.p):
                xinc = (maxx[feat] - minx[feat])/(self.numcut+1)
                self.xi[feat] = [
                    np.arange(1, (self.numcut)+1)*xinc + minx[feat]]
        
        # Set the terminal node hyperparameters
        self.tau = (self.rgy[1] - self.rgy[0])/(2*np.sqrt(self.ntree)*self.k)

        # Overwrite the hyperparameter settings when model mixing
        if self.modeltype == 9:
            if self.nsprior:
                self.tau = 1/(2*self.ntree*self.k)
                self.beta = 1/self.ntree
            else:
                self.tau = 1/(2*np.sqrt(self.ntree)*self.k)
                self.beta = 1/(2*self.ntree)
        else:
            self.beta = 0

        # Map for the overall sd default values
        osd = np.std(self.y_train, ddof = 1)
        osd_map = {1: 1, 2: 1, 3: 1, 4: osd, 5: osd, 6: 1, 7: 1, 8: osd, 9:osd} # ****** Think about this for model mixing
        
        # Overall sd update
        if (self.overallsd is None):
             print("Overwriting overallsd to agree with the model's default")
             self.overallsd = osd_map[self.modeltype]
        
        # overall lambda calibration
        self.overalllambda = self.overallsd**2
        
        # Birth and Death probability -- set product tree pbd to 0 for selected models 
        if (self.modeltype == 6) & (isinstance(self.pbd, float)):
            self.pbd = [self.pbd, 0]
        if self.modeltype in [4,9] and (isinstance(self.pbd, float)):
            self.pbd = [self.pbd, 0] 
        
        [self._update_h_args(arg) for arg in ["power", "base",
                                              "pbd", "pb", "stepwpert",
                                              "probchv", "minnumbot"]]
        
        # define the roots for the output files
        self.xroot = "x"
        self.yroot = "y"
        self.sroot = "s"
        self.chgvroot = "chgv"
        self.froot = "f"
        self.fsdroot = "fsd"
        self.wproot = "wpr"
        self.xiroot = "xi"
        # Check probit:
        if self.modeltype == 6:
            if self.ntreeh > 1:
                raise ValueError("ntreeh should be 1")
            if self.pbdh > 0:
                raise ValueError("pbdh should be 1")
        # Special quantity for merck_truncated:
        if (self.truncateds is None) & (self.modeltype == 8):
             miny = np.min(self.y_train)
             self.truncateds = (self.y_train == miny)
        if self.tc <= 1: 
             print("Setting tc to 2"); self.tc = 2
        # print((self.k, self.overallsd, self.overallnu, self.ntree, self.ntreeh))


    # Need to generalize -- this is only used in fit 
    def _write_config_file(self):
        """Create temp directory to write config and data files
        """
        f = tempfile.mkdtemp(prefix="openbtpy_")
        self.fpath = Path(f)
        run_params = [self.modeltype,
                      self.xroot, self.yroot, self.fmean_out,
                      self.ntree, self.ntreeh,
                      self.ndpost, self.nskip,
                      self.nadapt, self.adaptevery,
                      self.tau, self.beta,
                      self.overalllambda,
                      self.overallnu, self.base,
                      self.power, self.baseh, self.powerh,
                      self.tc, self.sroot, self.chgvroot,
                      self.froot, self.fsdroot, self.nsprior,
                      self.wproot, self.wtsprior,
                      self.pbd, self.pb, self.pbdh, self.pbh, self.stepwpert,
                      self.stepwperth,
                      self.probchv, self.probchvh, self.minnumbot,
                      self.minnumboth, self.printevery, self.xiroot, self.modelname,
                      self.summarystats]
        # print(run_params)
        self.configfile = Path(self.fpath / "config")
        with self.configfile.open("w") as tfile:
            for param in run_params:
                tfile.write(str(param)+"\n")
        # print(os.path.abspath(self.configfile))
        # sys.exit('Examining tmp file(s)') # The config file was correct when I looked at it manually.


    def __write_chunks(self, data, no_chunks, var, *args):
        if no_chunks == 0:
             print("Writing all data to one 'chunk'"); no_chunks = 1
        if (self.tc - int(self.tc) == 0):
             splitted_data = np.array_split(data, no_chunks)
        else:
             sys.exit('Fit: Invalid tc input - exiting process')   
        int_added = 0 if var in ["xp","fp","xw"] else 1

        # print(splitted_data)
        for i, ch in enumerate(splitted_data):
             # print(i); print(ch)
             np.savetxt(str(self.fpath / Path(self.__dict__[var+"root"] + str(i+int_added))),
               ch, fmt=args[0])


    def _write_data(self):
        splits = (self.n - 1) // (self.n/(self.tc)) # Should = tc - 1 as long as n >= tc
        # print("splits =", splits)
        self.__write_chunks(self.y_train, splits, "y", '%.7f')
        self.__write_chunks(np.transpose(self.X_train), splits, "x", '%.7f')
        self.__write_chunks(np.ones((self.n), dtype="int"),
                            splits, "s", '%.0f')
        print(self.fpath)
        if self.X_train.shape[0] == 1:
             print("1 x variable, so correlation = 1")
             np.savetxt(str(self.fpath / Path(self.chgvroot)), [1], fmt='%.7f')
        elif self.X_train.shape[0] == 2:
             print("2 x variables")
             np.savetxt(str(self.fpath / Path(self.chgvroot)),
                        [spearmanr(self.X_train, axis=1)[0]], fmt='%.7f')
        else:
             print("3+ x variables")
             np.savetxt(str(self.fpath / Path(self.chgvroot)),
                        spearmanr(self.X_train, axis=1)[0], fmt='%.7f')
             
        for k, v in self.xi.items():
            np.savetxt(
                str(self.fpath / Path(self.xiroot + str(k+1))), v, fmt='%.7f')
        
        # Write model mixing files
        if self.modeltype == 9:
            # F-hat matrix
            self.__write_chunks(self.F_train, splits, "f", '%.7f')
            # S-hat matrix when using nsprior
            if self.nsprior:
                self.__write_chunks(self.S_train, splits, "fsd", '%.7f')
            # Wts prior when passed in
            if self.wtsprior:
                np.savetxt(str(self.fpath / Path(self.wproot)), np.concatenate(self.betavec, self.tauvec),fmt='%.7f')
        
        # print(os.path.abspath(self.fpath))
        # sys.exit('Examining tmp file(s)') # The data files were correct:
        # For tc = 4: 1 chgv, 1 config, 3 s's, 3 x's, 3 y's, 1 xi (xi had the most data).


    def _run_model(self, train=True):
        cmd = "openbtcli" if train else "openbtpred"
        sp = subprocess.run(["mpirun", "-np", str(self.tc), cmd, str(self.fpath)],
                            stdin=subprocess.DEVNULL, capture_output=True)
        # print(sp)
        # A check:
        # if not(train):
        #     print(os.path.abspath(self.fpath)); sys.exit('Examining tmp file(s)')



    def predict(self, X, q_lower=0.025, q_upper=0.975, F = None, **kwargs):
        """
        Like with fit() X (the preds matrix) should be a numpy array, 
        but I added a couple of checks to allow a list to be accepted.
        """
        # Casting lists to arrays when needed
        if (type(X) == list):
            X = np.array(X)
            print("Completed list-to-numpy_array conversion for the preds. Be careful about row/column mixups!")
        if (len(X.shape) == 1): # If shape is (n, ), change it to (n, 1):
            X = X.reshape(len(X), 1) 
            # ^ Reshaping in case of weird 1-variable input (e.g. a 1D list)
        
        # Model mixing checks
        if self.modeltype == 9:
            if isinstance(F, list):
                F = np.array(F)
            if F is None:
                raise TypeError("No F matrix was provided for model mixing predictions.")
            if not F.shape[1] == self.nummodels:
                raise ValueError("Number of columns in F does not equal the number of models which were used in the training phase.")
            if not F.shape[0] == X.shape[0]:
                raise ValueError("Number of rows in F does not match the number of rows in X.")

        # Set control values
        self.p_test = X.shape[1]
        self.n_test = X.shape[0]
        self.q_lower = q_lower; self.q_upper = q_upper
        self.xproot = "xp"
        self.fproot = "fp"
        self.__write_chunks(X, (self.n_test) // (self.n_test/(self.tc)),
                            self.xproot,
                            '%.7f')
        
        # Write chunks for f in model mixing
        if self.modeltype == 9:
            self.__write_chunks(F, (self.n_test) // (self.n_test/(self.tc)),
                            self.fproot,
                            '%.7f')

        # Set and write config file
        self.configfile = Path(self.fpath / "config.pred")
        pred_params = [self.modelname, self.modeltype,
                       self.xiroot, self.xproot, self.fproot,
                       self.ndpost, self.ntree, self.ntreeh,
                       self.p_test, self.nummodels ,self.tc, self.fmean]
        # print(self.ntree); print(self.ntreeh)
        with self.configfile.open("w") as pfile:
            for param in pred_params:
                pfile.write(str(param)+"\n")
        self._run_model(train=False)
        self._read_in_preds()
        # New: make things a bit more like R, and save attributes to a fit object:
        res = {}
        res['mdraws'] = self.mdraws; res['sdraws'] = self.sdraws;
        res['mmean'] = self.mmean; res['smean'] = self.smean;
        res['msd'] = self.msd; res['ssd'] = self.ssd;
        res['m_5'] = self.m_5; res['s_5'] = self.s_5;
        res['m_lower'] = self.m_lower; res['s_lower'] = self.s_lower;
        res['m_upper'] = self.m_upper; res['s_upper'] = self.s_upper;
        res['q_lower'] = self.q_lower; res['q_upper'] = self.q_upper;
        res['x_test'] = X; res['modeltype'] = self.modeltype
        return res


    def _read_in_preds(self):
        mdraw_files = sorted(list(self.fpath.glob(self.modelname+".mdraws*")))
        sdraw_files = sorted(list(self.fpath.glob(self.modelname+".sdraws*")))
        mdraws = []
        for f in mdraw_files:
            read = open(f, "r"); lines = read.readlines()
            if lines[0] != '\n' and lines[1] != '\n': # If it's nonempty
                 mdraws.append(np.loadtxt(f))
        # print(mdraws[0].shape); print(len(mdraws))
        self.mdraws = np.concatenate(mdraws, axis=1) # Got rid of the transpose
        sdraws = []
        for f in sdraw_files:
            read = open(f, "r"); lines = read.readlines()
            if lines[0] != '\n' and lines[1] != '\n': # If it's nonempty
                 sdraws.append(np.loadtxt(f))
        # print(sdraws[0]); print(sdraws[0][0])
        # print(len(sdraws)); print(len(sdraws[0])); print(len(sdraws[0][0]))
        self.sdraws = np.concatenate(sdraws, axis=1) # Got rid of the transpose
        
        # New (added by me), since R returns arrays like these by default:
        # Calculate mmean and smean arrays, and related statistics
        self.mmean = np.empty(len(self.mdraws[0]))
        self.smean = np.empty(len(self.sdraws[0]))
        self.msd = np.empty(len(self.mdraws[0]))
        self.ssd = np.empty(len(self.mdraws[0]))
        self.m_5 = np.empty(len(self.mdraws[0]))
        self.s_5 = np.empty(len(self.mdraws[0]))
        self.m_lower = np.empty(len(self.mdraws[0]))
        self.s_lower = np.empty(len(self.sdraws[0]))
        self.m_upper = np.empty(len(self.mdraws[0]))
        self.s_upper = np.empty(len(self.sdraws[0]))
        for j in range(len(self.mdraws[0])):
             self.mmean[j] = np.mean(self.mdraws[:, j])
             self.smean[j] = np.mean(self.sdraws[:, j])
             self.msd[j] = np.std(self.mdraws[:, j], ddof = 1)
             self.ssd[j] = np.std(self.sdraws[:, j], ddof = 1)
             self.m_5[j] = np.quantile(self.mdraws[:, j], 0.50)
             self.s_5[j] = np.quantile(self.sdraws[:, j], 0.50)
             self.m_lower[j] = np.quantile(self.mdraws[:, j], self.q_lower)
             self.s_lower[j] = np.quantile(self.sdraws[:, j], self.q_lower)
             self.m_upper[j] = np.quantile(self.mdraws[:, j], self.q_upper)
             self.s_upper[j] = np.quantile(self.sdraws[:, j], self.q_upper)


    def clean_model(self):
        subprocess.run(f"rm -rf {str(self.fpath)}", shell=True)
               
#------------------------------------------------------------------------------------------
# Clark's functions (made from scratch, not edited from Zoltan's version):
    def _read_in_vartivity(self, q_lower, q_upper):
        vdraws_files = sorted(list(self.fpath.glob(self.modelname+".vdraws")))
        self.vdraws = np.array([])
        for f in vdraws_files:
            read = open(f, "r"); lines = read.readlines()
            if lines[0] != '\n' and lines[1] != '\n': # If it's nonempty
                 self.vdraws = np.append(self.vdraws, np.loadtxt(f))
        # self.vdraws[3] = 0.5 # to test transposing/ normalizing counts
        self.vdraws = self.vdraws.reshape(self.ndpost, self.p)
        # print(self.vdraws.shape); print(self.vdraws)
        # Normalize counts:
        colnorm = np.array([])
        for i in range(len(self.vdraws)): # should = ndpost in most cases
             colnorm = np.append(colnorm, self.vdraws[i].sum())
        idx = np.where(colnorm > 0)[0] # print(idx)
        # print(colnorm); print(idx)
        
        colnorm = colnorm.reshape(self.ndpost, 1) # Will always have 1 column b/c we summed
        # print(colnorm.shape); print(idx.shape)
        self.vdraws[idx] = self.vdraws[idx] / colnorm[idx]
        # print(self.vdraws[0:60]); print(self.vdraws.shape)
        
        self.mvdraws = np.empty(self.p)
        self.vdraws_sd = np.empty(self.p)
        self.vdraws_5 = np.empty(self.p)
        self.q_lower = q_lower; self.q_upper = q_upper
        self.vdraws_lower = np.empty(self.p)
        self.vdraws_upper = np.empty(self.p)

        for j in range(len(self.vdraws[0])): # (should = self.p)
             self.mvdraws[j] = np.mean(self.vdraws[:, j])
             self.vdraws_sd[j] = np.std(self.vdraws[:, j], ddof = 1)
             self.vdraws_5[j] = np.quantile(self.vdraws[:, j], 0.50)
             self.vdraws_lower[j] = np.quantile(self.vdraws[:, j], self.q_lower)
             self.vdraws_upper[j] = np.quantile(self.vdraws[:, j], self.q_upper)
        if (len(self.vdraws[0]) == 1): #  Make the output just a double, not a 2D array
             self.mvdraws = self.mvdraws[0]
             self.vdraws_sd = self.vdraws_sd[0]
             self.vdraws_5 = self.vdraws_5[0]
             self.vdraws_lower = self.vdraws_lower[0]
             self.vdraws_upper = self.vdraws_upper[0]
             
        # Now do everything again for the "h" version of all these quantities: 
        vdrawsh_files = sorted(list(self.fpath.glob(self.modelname+".vdrawsh")))
        self.vdrawsh = np.array([])
        for f in vdrawsh_files:
            self.vdrawsh = np.append(self.vdrawsh, np.loadtxt(f))
        self.vdrawsh = self.vdrawsh.reshape(self.ndpost, self.p)
        # Normalize counts:
        colnormh = np.array([])
        for i in range(len(self.vdrawsh)): # should = ndpost in most cases
             colnormh = np.append(colnormh, self.vdrawsh[i].sum())
        idxh = np.where(colnormh > 0)[0]
        colnormh = colnormh.reshape(self.ndpost, 1) # Will always have 1 column since we summed
        self.vdrawsh[idxh] = self.vdrawsh[idxh] / colnormh[idxh]
        
        self.mvdrawsh = np.empty(self.p)
        self.vdrawsh_sd = np.empty(self.p)
        self.vdrawsh_5 = np.empty(self.p)
        self.vdrawsh_lower = np.empty(self.p)
        self.vdrawsh_upper = np.empty(self.p)

        for j in range(len(self.vdrawsh[0])): # (should = self.p)
             self.mvdrawsh[j] = np.mean(self.vdrawsh[:, j])
             self.vdrawsh_sd[j] = np.std(self.vdrawsh[:, j], ddof = 1)
             self.vdrawsh_5[j] = np.percentile(self.vdrawsh[:, j], 0.50)
             self.vdrawsh_lower[j] = np.percentile(self.vdrawsh[:, j], self.q_lower)
             self.vdrawsh_upper[j] = np.percentile(self.vdrawsh[:, j], self.q_upper)
             
        if (len(self.vdrawsh[0]) == 1): #  Make the output just a double, not a 2D array
             self.mvdrawsh = self.mvdrawsh[0]
             self.vdrawsh_sd = self.vdrawsh_sd[0]
             self.vdrawsh_5 = self.vdrawsh_5[0]
             self.vdrawsh_lower = self.vdrawsh_lower[0]
             self.vdrawsh_upper = self.vdrawsh_upper[0]
         
             
    def vartivity(self, q_lower=0.025, q_upper=0.975):
        """Calculate and return variable activity information
        """
        # params (all are already set, actually)
        # self.p = len(self.xi[0][0])? # This definition is bad, but p is already defined in define_params()
        # Write to config file:
        vartivity_params = [self.modelname, self.ndpost, self.ntree,
                            self.ntreeh, self.p]
        self.configfile = Path(self.fpath / "config.vartivity")
        # print(vartivity_params)
        with self.configfile.open("w") as tfile:
            for param in vartivity_params:
                tfile.write(str(param)+"\n")
        # Run vartivity program  -- it's not actually parallel so no call to mpirun.
        # run_local = os.path.exists("openbtvartivity") # Doesn't matter, since the command is the same either way
        cmd = "openbtvartivity"
        sp = subprocess.run([cmd, str(self.fpath)],
                            stdin=subprocess.DEVNULL, capture_output=True)
        # print(sp)
        # Read in result (and set extra attributes like .5, .lower, .upper, etc.):
        self._read_in_vartivity(q_lower, q_upper)
        
        # Compile all the new attributes into something that will be saved as "fitv" when the function is called:
        res = {}
        res['vdraws'] = self.vdraws; res['vdrawsh'] = self.vdrawsh; 
        res['mvdraws'] = self.mvdraws; res['mvdrawsh'] = self.mvdrawsh; 
        res['vdraws_sd'] = self.vdraws_sd; res['vdrawsh_sd'] = self.vdrawsh_sd; 
        res['vdraws_5'] = self.vdraws_5; res['vdrawsh_5'] = self.vdrawsh_5; 
        res['vdraws_lower'] = self.vdraws_lower; res['vdrawsh_lower'] = self.vdrawsh_lower; 
        res['vdraws_upper'] = self.vdraws_upper; res['vdrawsh_upper'] = self.vdrawsh_upper; 
        res['q_lower'] = self.q_lower; res['q_upper'] = self.q_upper;
        res['modeltype'] = self.modeltype
        return res
   
    
   
     
    def _read_in_sobol(self, q_lower, q_upper):
        sobol_draws_files = sorted(list(self.fpath.glob(self.modelname+".sobol*")))
        # print(sobol_draws_files)
        self.so_draws = np.loadtxt(sobol_draws_files[0])
        for i in range(1, self.tc):
             read = open(sobol_draws_files[i], "r"); lines = read.readlines()
             if lines[0] != '\n' and lines[1] != '\n': # If it's nonempty
                  self.so_draws = np.vstack((self.so_draws,
                                  np.loadtxt(sobol_draws_files[i])))   
        # print(self.so_draws.shape); print(self.so_draws[0:10])
        labs_temp = list(itertools.combinations(range(1, self.p + 1), 2))
        labs = np.empty(len(labs_temp), dtype = '<U4')
        for i in range(len(labs_temp)):
             labs[i] =  ', '.join(map(str, labs_temp[i]))
        # print(self.so_draws.shape)
        ncol = self.so_draws.shape[1]; # nrow = self.so_draws.shape[0]
        draws = self.so_draws; p = self.p # ^ Shorthand to make the next lines shorter
        # All this is the same as R, but the beginning of the indices are shifted by
        # 1 since Python starts at index 0. Remember, Python omits the column at the 
        # end of the index, so the end index is actually the same as R!
        self.num_pairs = int(self.p*(self.p-1)/2)
        self.vidraws = draws[:, 0:p] # Columns 1-2 for p = 2
        self.vijdraws = draws[:, p:p+self.num_pairs] # Column 3 for p = 2
        self.tvidraws = draws[:, (ncol-1-p):(ncol-1)] # Columns 4 and 5 for p = 2
        self.vdraws = draws[:, ncol-1].reshape(self.ndpost, 1) # Column 6 for p = 2 (aLways last column)
        self.sidraws = self.vidraws / self.vdraws
        self.sijdraws = self.vijdraws / self.vdraws
        self.tsidraws = self.tvidraws / self.vdraws
        # ^ Colnames? If so, likely use list(range(1, p+1))
        # Compute a ton of sobol statistics:
        self.msi = np.empty(self.p)
        self.msi_sd = np.empty(self.p)
        self.si_5 = np.empty(self.p)
        self.q_lower = q_lower; self.q_upper = q_upper
        self.si_lower = np.empty(self.p)
        self.si_upper = np.empty(self.p)
        for j in range(len(self.sidraws[0])): # (should = self.p?)
             self.msi[j] = np.mean(self.sidraws[:, j])
             self.msi_sd[j] = np.std(self.sidraws[:, j], ddof = 1)
             self.si_5[j] = np.percentile(self.sidraws[:, j], 0.50)
             self.si_lower[j] = np.percentile(self.sidraws[:, j], self.q_lower)
             self.si_upper[j] = np.percentile(self.sidraws[:, j], self.q_upper)
             
        if (len(self.sidraws[0]) == 1): #  Make the output just a double, not a 2D array
             self.msi = self.msi[0]
             self.msi_sd = self.msi_sd[0]
             self.si_5 = self.si_5[0]
             self.si_lower = self.si_lower[0]
             self.si_upper = self.si_upper[0]
        # ^ Names? Nah   
        # Do this again for i,j:
        self.msij = np.empty(self.num_pairs)
        self.sij_sd = np.empty(self.num_pairs)
        self.sij_5 = np.empty(self.num_pairs)
        self.sij_lower = np.empty(self.num_pairs)
        self.sij_upper = np.empty(self.num_pairs)
        for j in range(len(self.sijdraws[0])): # (should = self.num_pairs?)
             self.msij[j] = np.mean(self.sijdraws[:, j])
             self.sij_sd[j] = np.std(self.sijdraws[:, j], ddof = 1)
             self.sij_5[j] = np.percentile(self.sijdraws[:, j], 0.50)
             self.sij_lower[j] = np.percentile(self.sijdraws[:, j], self.q_lower)
             self.sij_upper[j] = np.percentile(self.sijdraws[:, j], self.q_upper)
             
        if (len(self.sijdraws[0]) == 1): #  Make the output just a double, not a 2D array
             self.msij = self.msij[0]
             self.sij_sd = self.sij_sd[0]
             self.sij_5 = self.sij_5[0]
             self.sij_lower = self.sij_lower[0]
             self.sij_upper = self.sij_upper[0]   
        # ^ Names? Nah  
        # Do this again for t:
        self.mtsi = np.empty(self.p)
        self.tsi_sd = np.empty(self.p)
        self.tsi_5 = np.empty(self.p)
        self.tsi_lower = np.empty(self.p)
        self.tsi_upper = np.empty(self.p)
        for j in range(len(self.tsidraws[0])): # (should = self.p?)
             self.mtsi[j] = np.mean(self.tsidraws[:, j])
             self.tsi_sd[j] = np.std(self.tsidraws[:, j], ddof = 1)
             self.tsi_5[j] = np.percentile(self.tsidraws[:, j], 0.50)
             self.tsi_lower[j] = np.percentile(self.tsidraws[:, j], self.q_lower)
             self.tsi_upper[j] = np.percentile(self.tsidraws[:, j], self.q_upper)
             
        if (len(self.tsidraws[0]) == 1): #  Make the output just a double, not a 2D array
             self.mtsi = self.mtsi[0]
             self.tsi_sd = self.tsi_sd[0]
             self.tsi_5 = self.tsi_5[0]
             self.tsi_lower = self.tsi_lower[0]
             self.tsi_upper = self.tsi_upper[0] 
        # ^ Names? Nah    
       
        
    def sobol(self, cmdopt = 'serial', q_lower=0.025, q_upper=0.975, tc = 4):  
        """Calculate Sobol indices (more accurate than vartivity)
        """
        if (self.p <= 1 or (self.p - int(self.p) != 0)):
             sys.exit('Sobol: p (number of variables) must be 2 or more')
        # Write to config file:  
        sobol_params = [self.modelname, self.xiroot, self.ndpost, self.ntree,
                            self.ntreeh, self.p, self.minx, self.maxx, self.tc]
        self.configfile = Path(self.fpath / "config.sobol")
        # print("Directory for sobol calculations:", self.fpath) #print(sobol_params); 
        with self.configfile.open("w") as tfile:
            for param in sobol_params:
                if type(param) != str and type(param) != int: # Makes minx & maxx into writable quantities, not arrays
                     for item in param:
                          tfile.write(str(item)+"\n")
                else: tfile.write(str(param)+"\n")
        # Run sobol program: optional to use MPI.
        cmd = "openbtsobol"
        if(cmdopt == 'serial'):
             sp = subprocess.run([cmd, str(self.fpath)],
                            stdin=subprocess.DEVNULL, capture_output=True)
        elif(cmdopt == 'MPI'):
             sp = subprocess.run(["mpirun", "-np", str(tc), cmd, str(self.fpath)],
                            stdin=subprocess.DEVNULL, capture_output=True)
        else:
             sys.exit('Sobol: Invalid cmdopt (command option)')
        # print(sp)
        # Read in result (and set a bunch of extra attributes):
        self._read_in_sobol(q_lower, q_upper)
        
        # Compile all the new attributes into something that will be saved as "fits" when the function is called:
        res = {}
        # colnames(res$vidraws)=paste("V",1:p,sep="") # Implement this (and all the other colnames) if you use pandas
        # Set all of the self variables/attributes to res here:
        res['vidraws'] = self.vidraws; res['vijdraws'] = self.vijdraws;
        res['tvidraws'] = self.tvidraws; res['vdraws'] = self.vdraws;
        res['sidraws'] = self.sidraws; res['sijdraws'] = self.sijdraws;
        res['tsidraws'] = self.tsidraws;
        res['msi'] = self.msi; res['msi_sd'] = self.msi_sd; res['si_5'] = self.si_5;
        res['si_lower'] = self.si_lower; res['si_upper'] = self.si_upper;
        res['msij'] = self.msij; res['sij_sd'] = self.sij_sd; res['sij_5'] = self.sij_5;
        res['sij_lower'] = self.sij_lower; res['sij_upper'] = self.sij_upper;
        res['mtsi'] = self.mtsi; res['tsi_sd'] = self.tsi_sd; res['tsi_5'] = self.tsi_5;
        res['tsi_lower'] = self.tsi_lower; res['tsi_upper'] = self.tsi_upper;
        res['q_lower'] = self.q_lower; res['q_upper'] = self.q_upper;
        res['modeltype'] = self.modeltype; res['so_draws'] = self.so_draws
        return res
    

    def mixingwts(self, X, q_lower=0.025, q_upper=0.975):
        # Checks for proper inputs and convert lists to arrays
        if not self.modeltype == 9:
            raise TypeError("Cannot call openbt.mixingwts() method for openbt objects that are not modeltype = 'mixbart'")
        if isinstance(X,list):
            X = np.array(X)
        if (len(X.shape) == 1):
            X = X.reshape(len(X), 1)
        if not self.p == X.shape[1]:
            raise ValueError("The X array does not have the appropriate number of columns.")
        
        
        # Set control parameters
        self.xwroot = "xw"
        self.fitroot= ".fit"  # default, needed when considering the general class of model mixing problems -- revist this later
        self.q_lower = q_lower
        self.q_upper = q_upper

        # write out the config file
        # Set control values
        self.n_test = X.shape[0] # no need to set this as a class object like in predict function
        self.__write_chunks(X, (self.n_test) // (self.n_test/(self.tc)),
                            self.xwroot,
                            '%.7f')
        
        # Set and write config file
        self.configfile = Path(self.fpath / "config.mxwts")
        pred_params = [self.modelname, self.modeltype,
                       self.xiroot, self.xwroot, self.fitroot,
                       self.ndpost, self.ntree, self.ntreeh,
                       self.p, self.nummodels ,self.tc]
        # print(self.ntree); print(self.ntreeh)
        with self.configfile.open("w") as pfile:
            for param in pred_params:
                pfile.write(str(param)+"\n")
        # run the program
        cmd = "openbtmixingwts"
        sp = subprocess.run(["mpirun", "-np", str(self.tc), cmd, str(self.fpath)],
                            stdin=subprocess.DEVNULL, capture_output=True)
        self._read_in_wts()
        
        # New: make things a bit more like R, and save attributes to a fit object:
        res = {}
        res['wdraws'] = self.wdraws 
        res['wmean'] = self.wmean 
        res['wsd'] = self.wsd
        res['w_5'] = self.w_5
        res['w_lower'] = self.w_lower
        res['w_upper'] = self.w_upper
        res['q_lower'] = self.q_lower; res['q_upper'] = self.q_upper
        res['x_test'] = X; res['modeltype'] = self.modeltype
        return res

    def _read_in_wts(self):
        # Initialize the wdraws dictionary
        self.wdraws = {}        

        # Initialize summary statistic matrices for the wts
        self.wmean = np.empty((self.n_test,self.nummodels))
        self.wsd = np.empty((self.n_test,self.nummodels))
        self.w_5 = np.empty((self.n_test,self.nummodels))
        self.w_lower = np.empty((self.n_test,self.nummodels))
        self.w_upper = np.empty((self.n_test,self.nummodels))


        # Get the weight files
        for k in range(self.nummodels):
            wdraw_files = sorted(list(self.fpath.glob(self.modelname+".w"+str(k+1)+"draws*")))
            wdraws = []
            for f in wdraw_files:
                read = open(f, "r"); lines = read.readlines()
                if lines[0] != '\n' and lines[1] != '\n': # If it's nonempty
                    wdraws.append(np.loadtxt(f))

            # Store the wdraws array in the self.wdraws dictionary under the key wtname  
            wtname = "w" + str(k+1)
            self.wdraws[wtname] = np.concatenate(wdraws, axis=1) # Got rid of the transpose
            
            for j in range(len(self.wdraws[wtname][0])):
                self.wmean[j][k] = np.mean(self.wdraws[wtname][:, j])
                self.wsd[j][k] = np.std(self.wdraws[wtname][:, j], ddof = 1)
                self.w_5[j][k] = np.percentile(self.wdraws[wtname][:, j], 0.50)
                self.w_lower[j][k] = np.percentile(self.wdraws[wtname][:, j], self.q_lower)
                self.w_upper[j][k] = np.percentile(self.wdraws[wtname][:, j], self.q_upper)


    def set_wts_prior(self, betavec, tauvec):
        # Cast lists to np.arrays when needed
        if isinstance(betavec, list):
            betavec = np.array(betavec)
        if isinstance(tauvec, list):
            tauvec = np.array(tauvec)

        # Check lengths
        if not (len(tauvec) == self.nummodels and len(betavec) == self.nummodels):
            raise ValueError("Incorrect vector length for tauvec and/or betavec. Lengths must be equal to the number of models.")

        # Store the hyperparameters passed in 
        self.wtsprior = True
        self.betavec = betavec
        self.tauvec = tauvec


    # Save a posterior tree fit object (post) from the tmp working directory
    # into a local zip file given by [file].zip
    # If not file option specified, uses [model name].zip as the file.
    def obt_save(self, post = None, dirname = None, postname = 'post_PyData', est = False):
        if est == False:
            if(type(post) != dict): sys.exit("Invalid posterior fit.\n")
            if(dirname == None): dirname = post['modelname']
            if(dirname[-3:] != ".obt"): dirname = dirname + ".obt"
            import posixpath
            fname = posixpath.split(dirname)[1]
            files = sorted(list(self.fpath.glob("*"))) # Files to save (long names)
            from zipfile import ZipFile; import pickle; import subprocess
            with ZipFile(dirname, 'w') as myZip:
                # Save contents of the temp file (s1, x1, y1, etc.) to a zip folder:
                for i in range(len(files)):
                    myZip.write(files[i], posixpath.split(files[i])[1])
                print("Saved fit setup files to", dirname)
                fit_name = dirname[:-4]+'_'+postname
                with open(fit_name, 'wb') as f:
                    pickle.dump(post, f)
                myZip.write(fit_name, postname)
                subprocess.run(f"rm -f {fit_name}", shell=True)
                print("Saved fit object to", dirname)
            myZip.close()
        else: # (est == True), save the fit ESTIMATOR object (usually called m)
            if (dirname == None):
                dirname = post # b/c the user may have entered things in the wrong order
            if (dirname == None): sys.exit("Please specify directory name to save the estimator object.\n")
            from joblib import dump
            dump(self, dirname+'.joblib')
            print("Saved estimator object to", dirname)
        


#------------------------------------------------------
# Helper Functions (outside of class methods)
#------------------------------------------------------
# Load the posterior tree fit object (doesn't load any files like s1, x1, y1, etc):
def obt_load(dirname = None, postname = 'post_PyData', est = False):  
    if est == False:
        if(dirname[-3:] != ".obt"): dirname = dirname + ".obt"
        import pickle; from zipfile import ZipFile
        with ZipFile(dirname, 'r') as myZip:
            # print(myZip.namelist())
            with myZip.open(postname) as myfile:
                loaded_obj = pickle.load(myfile)
        myZip.close()
    else: # (est == True), load the fit ESTIMATOR object (usually called m)
        if (dirname == None): sys.exit("Please specify directory name to load the estimator object.\n")
        if(dirname[-6:] != ".joblib"): dirname = dirname + ".joblib"
        from joblib import load
        loaded_obj = load(dirname)
    return(loaded_obj)


#------------------------------------------------------
# Future Work --- OpenbtMixing Class, inherits from Openbt
#------------------------------------------------------


