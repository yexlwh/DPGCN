
import numpy as np
from numpy.matlib import repmat

def vdpmm_init(testData,K):
    params={}
    num,dim=testData.shape
    gammas = np.random.rand(num,K)
    temp=repmat(np.sum(gammas,axis=1),K,1)
    temp1=temp.transpose()
    gammas = gammas /temp1
    params['eq_alpha'] = 50
    params['beta'] = np.zeros((K,1))
    params['a'] = np.zeros((K,1))
    params['meanN'] = np.zeros((dim,K))
    params['B'] = np.zeros((dim,dim,K))
    params['sigma'] = np.zeros((dim,dim,K))
    params['mean'] = np.zeros((K,dim))
    params['g'] = np.zeros((K,2))
    params['ll'] = -np.inf
    return params,gammas
