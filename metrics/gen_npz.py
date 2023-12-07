import numpy as np


mu = np.load('npy_files/mu_test.npy')
sigma = np.load('npy_files/sigma_test.npy')

np.savez('test_stats.npz', mu=mu, sigma=sigma)
