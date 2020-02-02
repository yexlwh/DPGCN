import numpy as np
from numpy.matlib import repmat
import scipy.io as sio

def vdpmm_init_load(testData,K):
    dataInput=sio.loadmat('init.mat')
    paramsTemp=dataInput['params']
    params={}
    num,dim=testData.shape

    params['eq_alpha'] = 50
    params['beta'] = np.zeros((K,1))
    params['a'] = np.zeros((K,1))
    params['meanN'] = np.zeros((dim,K))
    params['B'] = np.zeros((dim,dim,K))
    params['sigma'] = np.zeros((dim,dim,K))
    params['mean'] = np.zeros((K,dim))
    params['g'] = np.zeros((K,2))
    params['ll'] = -np.inf
    
    gammas=dataInput['gammas']
    return params,gammas
