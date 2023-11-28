import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


df = pd.read_csv('Prices.csv')

# 1)
x1 = df['Speed'].values
x2 = np.log(df['HardDrive'].values)
y = df['Price'].values


with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = alpha + beta1 * x1 + beta2 * x2

    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(2000, tune=500, cores=1)

az.plot_posterior(trace, var_names=['alpha', 'beta1', 'beta2', 'sigma'])

# 2)

az.plot_posterior(trace, var_names=['beta1'], hdi_prob=0.95, figsize=(8, 4))
plt.title('Posterior Distribution of beta1 with 95% HDI')
plt.show()

az.plot_posterior(trace, var_names=['beta2'], hdi_prob=0.95, figsize=(8, 4))
plt.title('Posterior Distribution of beta2 with 95% HDI')
plt.show()

# 4)

# Valorile de care este interesat consumatorul
new_x1 = 33
new_x2 = np.log(540)

with model:
    new_mu = pm.Normal('new_mu', mu=alpha + beta1 * new_x1 + beta2 * new_x2, sd=sigma)
    new_trace = pm.sample_posterior_predictive(trace, var_names=['new_mu'], samples=5000)

hdi_90 = az.hdi(new_trace['new_mu'], hdi_prob=0.9)

az.plot_posterior(new_trace, hdi_prob=0.9, figsize=(8, 4))
plt.title('Interval HDI 90% pentru pretul asteptat')
plt.show()
