import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
import statsmodels.api as sm

from animate import animate
from rossler_map import RosslerMap

def plot_local_maxima(z):
    indices = argrelextrema(z, np.greater)[0]
    xindices = indices[:-1]
    yindices = indices[1:]
    plt.figure()
    plt.title('Florent plot')
    plt.plot(z[xindices], z[yindices], 'x')
    plt.xlabel('previous z')
    plt.ylabel('next z')
    plt.show()


    
def plot_things(y, yhat):

    plt.figure()
    plt.title('Time series: y vs timestep')
    plt.xlabel('timestep')
    plt.ylabel('y')
    plt.plot(y)
    plt.plot(yhat)
    plt.legend(['y', '$\hat{y}$'])

    plt.figure()
    plt.title('y')
    plt.hist(y)

    plt.figure()
    plt.title('$\hat{y}$')
    plt.hist(yhat)


    sm.graphics.tsa.plot_pacf(y, lags=100, title='PACF y')
    sm.graphics.tsa.plot_pacf(yhat, lags=100, title='PACF $\hat{y}$')

    plt.show()


if __name__ == '__main__':
    realtraj = np.load('real_traj.npy')
    y = realtraj[:, 1] 

    yhat = np.load('y_traj.npy')
    ourtraj = np.load('our_traj.npy')

    plot_things(y, yhat)
    animate(realtraj.T, ourtraj.T, realtraj.shape[0], interval=1)


