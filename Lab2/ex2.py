import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

alpha_values = [4, 4, 5, 5]
lambda_values = [3, 2, 2, 3]
latency_lambda = 4
probabilities = [0.25, 0.25, 0.30, 0.20]

no_samples = 10000

X = np.zeros(no_samples)

for i in range(no_samples):
    server_index = np.random.choice(len(probabilities), p=probabilities)
    server_alpha = alpha_values[server_index]
    server_lambda = lambda_values[server_index]
    server_time = stats.gamma.rvs(server_alpha, scale=1/server_lambda)
    latency_time = stats.expon.rvs(scale=1/latency_lambda)
    X[i] = server_time + latency_time

probability = np.count_nonzero(X > 3) / 10000

print(f"Probabilitatea ca X > 3 milisecunde: {probability * 100} %")

# plt.hist(X, bins=50, density=True, alpha=0.6, color='b', label='Densitatea lui X')
# plt.xlabel('Timp de servire (X)')
# plt.ylabel('Densitate')
# plt.legend(loc='upper right')
# plt.title('Densitatea distribuției lui X')
az.plot_posterior(X)
plt.show()
