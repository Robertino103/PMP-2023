import pandas as pd
import pymc as pm
import numpy as np
import arviz as az

try:
    np.distutils.__config__.blas_opt_info = np.distutils.__config__.blas_ilp64_opt_info
except Exception:
    pass

data = pd.read_csv('trafic.csv', usecols=['minut', 'nr. masini'])

changes = [7 * 60, 8 * 60, 16 * 60, 19 * 60]

# Sort the changes vector for convenience
changes = sorted(changes)

basic_model = pm.Model()

with basic_model:
    lambda_pieces = []
    for i in range(len(changes) + 1):
        if i == 0:
            lambda_i = pm.Uniform(f'lambda_{i}', lower=0, upper=100)
        else:
            lambda_i = pm.Uniform(f'lambda_{i}', lower=0, upper=100)
        lambda_pieces.append(lambda_i)

    lambda_values = [pm.math.switch(data['minut'] > change, lambda_pieces[i], lambda_pieces[i + 1])
                     for i, change in enumerate(changes)]

    lambda_combined = pm.math.switch(data['minut'] >= changes[-1], lambda_pieces[-1], lambda_values[-1])

    Y_obs = pm.Poisson('Y_obs_combined', mu=lambda_combined, observed=data['nr. masini'])

    trace = pm.sample(500, cores=1)

    az.plot_posterior(trace, var_names=['lambda_', 'lambda_0', 'lambda_1', 'lambda_2', 'lambda_3'])

summary = az.summary(trace)
print(summary)
