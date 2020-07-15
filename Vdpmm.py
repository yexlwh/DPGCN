############
from dpgcn.dp_init import *
from  dpgcn.ini import *
from dpgcn.vdpmm_maximize import *
from dpgcn.vdpmm_expectation import *
def dpmm(testData):
    K=50
    infinite=1
    verbose=1
    maxits=160
    minits=10
    eps=0.01
    params,gammas = vdpmm_init(testData,K)
    numits = 2;
    score = -np.inf;
    score_change = np.inf;
    cho=0;

    while numits < maxits:
        print('run',numits)
        params = vdpmm_maximizeC(testData,params,gammas);
        gammas=vdpmm_expectationC(testData,params)
        numits=numits+1
    return gammas
