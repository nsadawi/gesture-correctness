# lots of these functions are adapted from free online sources
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

def rand_list(start, end, num): 
    """ Generate a random list within start/end range incl\
    Makes sure all elements are unique"""
    res = [] 
    while(1):
        for j in range(num): 
            res.append(random.randint(start, end))
        res = list(set(res))
        if len(res) == num:
            break
        else:
            res = []
  
    return res 

def normalize(data):
    """ Normalize data to range [0,1]"""
    return (data - np.min(data, axis=0)[None,:,:]) / (np.max(data, axis=0)[None,:,:] - np.min(data, axis=0)[None,:,:])

def normalize_df(df,min_val = 0, max_val = 1):
    """transform values in each column in df to the given value range"""
    columns = df.columns
    scaler = MinMaxScaler(feature_range=(min_val, max_val))
    return (pd.DataFrame(scaler.fit_transform(df),columns=columns))

def standardize(data, mean=0, std = 1, train = True):
    """ Standardize data .. compute z-score """
    if train:
        mean = np.mean(data, axis=0)[None,:,:]
        std  = np.std(data, axis=0)[None,:,:]
        data = (data - mean) / std
        return data, mean, std
    else:
        data = (data - mean) / std
        return data,None,None

def zero_mean(data, mean=0, train = True):
    """ Standardize data .. compute z-score """
    if train:
        mean = np.mean(data, axis=0)[None,:,:]
        return (data - mean), mean
    else:
        return (data - mean), None


def one_hot(labels):
    """ One-hot encoding """
    y = np.asarray(pd.get_dummies(labels), dtype = np.int8)
    return y


# from https://stackoverflow.com/a/4602224
def unison_shuffled_copies(a, b):
    """ shuffles two matrices in unison """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_batches(X, y, batch_size = 16, shuffle = False):
    """ Return a generator for batches """
    if shuffle:
        X,y = unison_shuffled_copies(X, y)

    if batch_size > len(X):
        batch_size = len(X)
    
    n_batches = len(X) // batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]

def get_random_batches(X, y, num_batches = 25, batch_size = 25):
    """ Return a generator for batches """
    # Loop over batches and yield
    for b in range(0, num_batches):
        start = np.random.randint(0, high=len(X)-batch_size)
        yield X[start:start+batch_size], y[start:start+batch_size]

def windows(data, size):
    """ Return a generator for overlapping windows """
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += size # no overlap between windows
        
def segment_signal(data, window_size, num_features):
    """ Segment data into overlapping windows """
    segments = np.empty((0,window_size,num_features))
    labels = np.empty((0))
    for (start, end) in windows(data["Class"], window_size):
        ls = []
        for i in range(0,num_features): 
            ls.append(data["f_"+str(i)][start:end])
        #x = data["x-axis"][start:end]
        #y = data["y-axis"][start:end]
        #z = data["z-axis"][start:end]
        #print((start, end))
        if(len(data["Class"][start:end]) == window_size):
            #print((start, end))
            segments = np.vstack([segments,np.dstack(ls)])
            labels = np.append(labels,stats.mode(data["Class"][start:end])[0][0])
    return segments, labels

def get_df_chunks(X, num_chunks):
    """ Return a generator for batches """
    assert num_chunks <= len(X), "num_chunks must be <= len(df)"
    chunk_size = len(X) // num_chunks  
    # Loop over batches and yield
    for b in range(0, num_chunks):
        if b == num_chunks - 1:
            yield X[b*chunk_size:]
        else:
            yield X[b*chunk_size:(b*chunk_size)+chunk_size]

def normalize_move(df, num_chunks, mean=True):
    """downsample a df to a certain number of rows, take mean or median"""
    dd = pd.DataFrame()
    if mean:# use mean of each chunk
        for dfx in get_df_chunks(df,num_chunks):
            dd = dd.append(dfx.mean(), ignore_index=True)
    else:# use median of each chunk        
        for dfx in get_df_chunks(df,num_chunks):
            dd = dd.append(dfx.median(), ignore_index=True)
    return dd

def mean_centre_data(df):
    """mean centre a dataframe"""
    return df - df.mean()