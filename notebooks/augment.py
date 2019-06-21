## Functions for Time Series Data Augmentation
## Some are modified a bit
## Main source:
## https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
    
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
import numpy as np

def DA_Jitter(X, sigma=0.001):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise

def DA_Scaling(X, sigma=0.001):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    #print(scalingFactor)
    return X*myNoise


# Hyperparameters : nPerm = # of segments to permute
# minSegLength = allowable minimum length for each segment
def DA_Permutation(X, nPerm=4, minSegLength=5):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)


## This example using cubic splice is not the best approach to generate random curves. 
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs = [CubicSpline(xx[:,i], yy[:,i]) for i in range(0, xx.shape[1])]
    return np.array([cs_element(x_range) for cs_element in cs]).transpose()

## This example using cubic splice is not the best approach to generate random curves. 
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
# only generate 3 randon curves
def GenerateRandomCurves3(X, num_cols = 3, sigma=0.05, knot=4):
    xx = (np.ones((num_cols,1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, num_cols))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    cs_z = CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()

def DistortTimesteps3(X, sigma=0.05):
    tt = GenerateRandomCurves3(X) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    #print(tt_cum, '----')
    return tt_cum

def DA_TimeWarp3(X, sigma=0.05):
    tt_new = DistortTimesteps3(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    for i in range(0,X.shape[1], 3):
        X_new[:,i] = np.interp(x_range, tt_new[:,0], X[:,i])
        X_new[:,i+1] = np.interp(x_range, tt_new[:,1], X[:,i+1])
        X_new[:,i+2] = np.interp(x_range, tt_new[:,2], X[:,i+2])
    return X_new

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = []
    for i in range(0,tt_cum.shape[1]):
        t_scale.append((X.shape[0]-1)/tt_cum[-1,i])

    for i in range(0,tt_cum.shape[1]):
        tt_cum[:,i] = tt_cum[:,i]*t_scale[i]
        
    return tt_cum

def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    print(tt_new.shape)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    for i in range(X.shape[1]):
        X_new[:,i] = np.interp(x_range, tt_new[:,i], X[:,i])
    #X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
    #X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
    return X_new

def DA_MagWarp(X, sigma=0.2):
    return X * GenerateRandomCurves(X, sigma)


def DA_Rotation(X):
    #np.random.seed(0)
    assert X.shape[1] % 3 == 0, "No of columns should be divisible by 3!"
    X_new = np.zeros(X.shape)
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    #print(angle)
    # this is to rotate all triples using same rotation matrix
    rotation_matrix = axangle2mat(axis[0:3],angle)
    #print(rotation_matrix)
    for i in range(0,X.shape[1], 3):
        #rotation_matrix = axangle2mat(axis[i:i+3],angle)
        #print(rotation_matrix)
        X_new[:,i:i+3] = np.matmul(X[:,i:i+3] , rotation_matrix)
    return X_new