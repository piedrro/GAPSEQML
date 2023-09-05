import numpy as np
from tqdm import tqdm
import random


def jittering(x, sigma=0.03):
    
    x = x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    
    return x

def scaling(x, sigma=0.1):
    
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0]))
    
    x = np.multiply(x, factor)
    
    return x

def rolling(x):

    start = np.random.randint(0, len(x))
    
    x = np.roll(x, start,0)
    
    return x
    
def slicing(x, slice_range = [95,100]):
    
    slice_percent = np.random.randint(slice_range[0],slice_range[1])
    
    slice_length = int((len(x)/100)*slice_percent)
    
    x = random.sample(list(x), slice_length)
    
    x = np.interp(np.linspace(0, len(x) - 1, num=1200), np.arange(len(x)), x)
    
    return x

    
def flipping(X):
    
    flip = bool(random.getrandbits(1))
    
    if flip:
    
        X = np.flip(X)
    
    return X
