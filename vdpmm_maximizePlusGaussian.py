#########
import numpy as np
from numpy.matlib import repmat

def vdpmm_maximizePlusGaussian(data,params,gammas):
    D = data.shape[1];
    N =data.shape[0];
    K = (params['a']).shape[0];
    a0 = D;
    beta0 = 1;
    mean0 = np.mean(data,axis=0);
    B0 = .1 * D * np.cov(data.T);

    #convenience variables first
    Ns = np.sum(gammas,axis=0) + 1e-10;
    mus = np.zeros((K,D))
    sigs = np.zeros((D,D,K))
    mus = np.dot(gammas.T , data) /(repmat(Ns,D,1).T)
    for i in range(K):
        diff0 = data - repmat(mus[i,:],N,1);
        diff1 = repmat(np.sqrt(gammas[:,i]),D,1).T * diff0;
        sigs[:,:,i] = np.dot(diff1.T , diff1);

    #now the estimates for the variational parameters
    params['g'][:,0] = 1 + np.sum(gammas,axis=0);
    #g_{s,2} = Eq[alpha] + sum_n sum_{j=s+1} gamma_j^n
    temp1=(params['eq_alpha'] +np.flipud(np.cumsum(np.flipud(np.sum(gammas,0)))) - np.sum(gammas,0))
    params['g'][:,1] = temp1;
    params['beta'] = Ns + beta0;
    params['a'] = Ns + a0;
    params['mean'] = (repmat(Ns,D,1).T * mus + beta0 * repmat(mean0,Ns.shape[0],1)) / (repmat(Ns.T + beta0,D,1).T)

    for i in range(K):
        diff = mus[i,:] - mean0
        params['B'][:,:,i] = sigs[:,:,i] + Ns[i] * beta0 * np.dot(diff,diff.T) / (Ns[i]+beta0) + B0
    return params
