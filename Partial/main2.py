import numpy as np
from scipy.stats import norm
import arviz as az
import matplotlib.pyplot as plt


# Generam 200 de timpi medii folosind mle:
np.random.seed(42)
wait_times = np.random.uniform(low=0, high=20, size=200)
plt.hist(wait_times)
plt.show()

import pymc as pm

# Definim modelul în PyMC
with pm.Model() as model:
    # Alegem distributii a priori pentru parametrii medie si deviatie standard
    mean_prior = pm.Normal('mean', mu=10, sigma=5)
    std_prior = pm.HalfNormal('std', sigma=2)

    # Definim distributia de observatii
    likelihood = pm.Normal('likelihood', mu=mean_prior, sigma=std_prior, observed=wait_times)


# Estimarea distributiei a posteriori
with model:
    trace = pm.sample(200, tune=100, cores=1)

az.plot_trace(trace)
plt.show()

az.plot_posterior(trace['mean'])
plt.show()

az.plot_posterior(trace['std'])
plt.show()
