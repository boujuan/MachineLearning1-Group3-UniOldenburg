import numpy as np 
import matplotlib.pyplot as plt 


# %% Generate Dataset
N = 100
mu = 20
sigma = 2
N_bins = 5
norm_dist = np.random.normal(mu, sigma, N)


x = np.linspace(np.floor(np.min(norm_dist)), np.ceil(np.max(norm_dist)), N_bins)
Delta = (np.diff(x))[0]

norm_dist_sampl = []

for i in range(len(x)-1):
    
    a = (norm_dist >= x[i])
    b = (norm_dist < x[i+1])
    
    ab = [a[i] and b[i] for i in range(len(a))]
    
    n_i = len(np.where(ab)[0])
    
    norm_dist_sampl.append(n_i/(N*Delta))

plt.scatter(norm_dist, [0]*N, marker=".")
plt.plot(x[:-1]+Delta/2, norm_dist_sampl)

print(np.sum(norm_dist_sampl)*Delta)

# %% Max Likelyhood parameter

mu_hat = (1/N) * np.sum(norm_dist)
var_hat = (1/N) * np.sum((norm_dist - mu_hat)**2)

print(mu_hat, var_hat)