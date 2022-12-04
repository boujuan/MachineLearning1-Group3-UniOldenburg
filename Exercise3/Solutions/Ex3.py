#%% Import

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm


#%% 1B

Theta = pd.DataFrame(index = ["c=1", "c=2"], columns=["pi", "mu", "sigma"])

Theta["pi"] = [0.4, 0.6]
Theta["mu"] = [-2, 4]
Theta["sigma"] = [1, 10]


def Gaussian(x, mu, sigma):
    return 1/np.sqrt(2*np.pi*sigma) * np.exp(-(x-mu)**2/(2*sigma))


    
x = np.linspace(-10, 20, 250)
dx = x[1] - x[0]
p_x = Theta.loc["c=1", "pi"] * Gaussian(x, Theta.loc["c=1", "mu"], Theta.loc["c=1", "sigma"]) +  Theta.loc["c=2", "pi"] * Gaussian(x, Theta.loc["c=2", "mu"], Theta.loc["c=2", "sigma"])
    
plt.plot(x, p_x*dx)
plt.xlabel("x")
plt.ylabel("p(x|$\Theta$)")
plt.savefig("/home/johannes/Dokumente/Uni/Ma_3rd Semester/Unsupervised_ML/Homework/Ex3_1B.png")
plt.show()

# %% 1C
N=10000

sample = np.random.choice(x, p=p_x/np.sum(p_x), size=N)
plt.scatter(sample, [0]*N, marker=".", s = 0.5)
plt.xlabel("x")
plt.savefig("/home/johannes/Dokumente/Uni/Ma_3rd Semester/Unsupervised_ML/Homework/Ex_3_1_C.png")
plt.show()

# %% 1D-F

# Histogram via Sampling formula

edges = np.linspace(-10, 20,7)
p_x_i = []
n_i = []

for idx, edge_up in enumerate(edges[1:]):
    edge_low = edges[idx]
    
    n_i.append(len([i for i in sample if edge_low<=i<edge_up]))
    d_bins = edge_up-edge_low
    
    p_x_i.append(n_i[-1] / (N * d_bins) )
    


center = (edges[:-1] + edges[1:])/2

# D
plt.bar(center, n_i)
# plt.hist(sample, bins=50, range=[-10,20])
plt.xlabel("x")
plt.ylabel("p(x|$\Theta$)")
plt.savefig("/home/johannes/Dokumente/Uni/Ma_3rd Semester/Unsupervised_ML/Homework/Ex_3_1_D.png")
plt.show()


#E, F
# plt.hist(sample, bins=50, range=[-10,20], density=True)
plt.bar(center, p_x_i)
plt.xlabel("x")
plt.ylabel("p(x|$\Theta$)")
plt.savefig("/home/johannes/Dokumente/Uni/Ma_3rd Semester/Unsupervised_ML/Homework/Ex_3_1_E.png")
plt.plot(x, p_x, c="red")
plt.savefig("/home/johannes/Dokumente/Uni/Ma_3rd Semester/Unsupervised_ML/Homework/Ex_3_1_F.png")
plt.show()

# %% 1G-H

p_x = Theta.loc["c=1", "pi"] * Gaussian(sample, Theta.loc["c=1", "mu"], Theta.loc["c=1", "sigma"]) +  Theta.loc["c=2", "pi"] * Gaussian(sample, Theta.loc["c=2", "mu"], Theta.loc["c=2", "sigma"])

Log_l = np.sum(np.log(p_x))

print(Log_l)


Theta2 = pd.DataFrame(index = ["c=1", "c=2"], columns=["pi", "mu", "sigma"])

Theta2["pi"] = [0.4, 0.6]
Theta2["mu"] = [-1, 5]
Theta2["sigma"] = [1, 7]

# pi1 = np.linspace(0,1,10)
# pi2 = 1-pi1

mu1 = np.linspace(-5,5,100)
mu2 = np.linspace(0,10,100)
ll = pd.DataFrame(index = mu1, columns=mu2)

for m in mu1:
    for m2 in mu2:
        p_x_m = Theta.loc["c=1", "pi"] * Gaussian(sample, m, Theta.loc["c=1", "sigma"]) +  Theta.loc["c=2", "pi"] * Gaussian(sample, m2, Theta.loc["c=2", "sigma"])
        ll.loc[m, m2] = np.sum(np.log(p_x_m))
        # ll.append(np.sum(np.log(p_x_m)))
    print(m)
    
    
mm1, mm2 = np.meshgrid(mu1, mu2)
LL = ll.to_numpy(dtype="float64")

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=[7.5,7.5])

ax.plot_surface(mm1, mm2, LL, cmap = cm.coolwarm,  linewidth=0, antialiased=False)
ax.set_xlabel("$\mu_1$")
ax.set_ylabel("$\mu_2$")
ax.set_zlabel("$\mathcal{L}(\Theta)$")

ax.scatter(-2, 4, np.max(LL)+10, marker="x")   


fig.savefig("/home/johannes/Dokumente/Uni/Ma_3rd Semester/Unsupervised_ML/Homework/Ex_3_LL_plot.png")