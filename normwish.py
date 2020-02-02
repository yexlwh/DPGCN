import numpy as np
import scipy.special as ssp
#from numpy.matlib import repmat

def normwish(x,mu,lambda1,a,B):
    print("hello world")
##    N,D = x.shape
##    Gamma_bar = 0.5 * a * np.linalg.inv(B)
##    d = x - np.squeeze(repmat(mu,N,1));
##    logprob = -.5 * D * np.log(2*pi) - .5 * (np.linalg.slogdet(.5*B))[1] +\
##    .5 * np.sum(ssp.psi(0.5 * (a + 1 - np.arange(D)))) - 0.5 * D / lambda1- \
##    np.sum((d * Gamma_bar)*d,2);
