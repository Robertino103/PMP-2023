import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

lambda1 = 4.0
lambda2 = 6.0
p1 = 0.4
p2 = 0.6
no_samples = 10000

num_samples_mechanic1 = np.random.binomial(no_samples, p=p1)
num_samples_mechanic2 = no_samples - num_samples_mechanic1

X1 = stats.expon(scale=1/lambda1).rvs(size=num_samples_mechanic1)
X2 = stats.expon(scale=1/lambda2).rvs(size=num_samples_mechanic2)
X = np.concatenate([X1, X2])

print("Media lui X:", np.mean(X1) * p1 + np.mean(X2) * p2)
print("Deviatia standard a lui X:", np.std(X))

# plt.hist(X, bins=50, density=True, alpha=0.6, color='b', label='Densitatea lui X')
# plt.xlabel('Timp de servire (X)')
# plt.ylabel('Densitate')
# plt.legend(loc='upper right')
# plt.title('Densitatea distribuției lui X')
az.plot_posterior(X)
plt.show()
