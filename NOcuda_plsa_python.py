import sys, time
import numpy as np
import pylab
pylab.seed(0)
from numpy import zeros, int8, log
import scipy.io as sio

def EStep():
    for i in range(0, N): # por cada pixel
        for j in range(0, M): # por cada banda
            denominator = 0;
            for k in range(0, K): # por cada endmember
                p[i, j, k] = theta[k, j] * lamda[i, k];
                denominator += p[i, j, k];
            if denominator == 0:
                for k in range(0, K):
                    p[i, j, k] = 0;
            else:
                for k in range(0, K):
                    p[i, j, k] /= denominator;

def MStep():
    # update theta
    for k in range(0, K): # por cada endmember
        denominator = 0
        for j in range(0, M): # de cada banda
            theta[k, j] = 0
            for i in range(0, N): # calcular thetas para cada pixel
                valueaux = X[i, j] * p[i, j, k] - (regularization1 / float(M))
                if valueaux > 0:
                    theta[k, j] += valueaux
            denominator += theta[k, j]
        if denominator == 0:
            for j in range(0, M):
                theta[k, j] = 1.0 / M
        else:
            for j in range(0, M):
                theta[k, j] /= denominator
    # update lamda
    for i in range(0, N):
        for k in range(0, K):
            lamda[i, k] = 0
            denominator = 0
            for j in range(0, M):
                valueaux = X[i, j] * p[i, j, k] - (regularization2 / float(K))
                if valueaux > 0:
                    lamda[i, k] += valueaux
                denominator += X[i, j];
            if denominator == 0:
                lamda[i, k] = 1.0 / K
            else:
                lamda[i, k] /= denominator

# calculate the
def LogLikelihood(): # calcular  log likelihood del modelo
    loglikelihood = 0
    for i in range(0, N): # por cada pixel
        for j in range(0, M): # de cada banda
            tmp = 0
            for k in range(0, K): # por cada endmember
                tmp += theta[k, j] * lamda[i, k] # calculo su probabilidad
            if tmp > 0:
                loglikelihood += X[i, j] * log(tmp)
    return loglikelihood




########################################################################################
##########################     PARAMETROS      #########################################
########################################################################################
K = 4    # numero de endmembers
maxIteration = 1
threshold = 10e-6
step2show = 1
step2save = 10
regularization1 = 0 # regularizador1
regularization2 = 0 # regularizador2
path_image = "inputs/samson_1.mat"
########################################################################################
########################################################################################
########################################################################################

image = sio.loadmat(path_image)['V'].T
nRow  = sio.loadmat(path_image)['nRow'][0][0]
nCol  = sio.loadmat(path_image)['nCol'][0][0]
image = np.transpose(image.reshape(nCol, nRow, image.shape[1]), (1,0,2))
shape_image = image.shape
N = shape_image[0] * shape_image[1] # num pixeles
M = shape_image[2] # num bands
X = image.reshape(shape_image[0] * shape_image[1], shape_image[2])

# lamda[i, j] : p(zj|di)
lamda = pylab.random([N, K]) # abundances
# theta[i, j] : p(wj|zi)
theta = pylab.random([K, M]) # endmembers
# p[i, j, k] : p(zk|di,wj)
p = zeros([N, M, K]) # posterior

# normalizacion de parametros lambda y theta
for i in range(0, N):
    normalization = sum(lamda[i, :])
    for j in range(0, K):
        lamda[i, j] /= normalization
for i in range(0, K):
    normalization = sum(theta[i, :])
    for j in range(0, M):
        theta[i, j] /= normalization

total_time = 0
for i in range(maxIteration):
    tic = time.time()
    EStep()
    MStep()
    toc = time.time()
    total_time += (toc - tic)
    print("CPU VERSION: Time epoch", np.round(toc-tic, 2), "Expected time", np.round((toc-tic)*100, 2))
    print("CUDA Version is", np.round((toc-tic)/0.01, 2), "more faster")
