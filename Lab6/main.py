from multiprocessing import freeze_support

import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

if __name__ == '__main__':
    freeze_support()

    fig, axes = plt.subplots(len(Y_values), len(theta_values), figsize=(12, 8))  # subploturi pt afisaj in o singura fereastra

    for i, Y in enumerate(Y_values):
        for j, theta in enumerate(theta_values):

            with pm.Model() as model:
                n = pm.Poisson("n", mu=10)  # distributia n a priori

                Y_observed = pm.Binomial(
                    f"Y_observed:{Y}{theta}", n=n, p=theta, observed=Y
                )

                trace = pm.sample(2000, tune=1000, cores=1)

                az.plot_posterior(
                    trace,
                    var_names=["n"],
                    point_estimate="mean",
                    ax=axes[i, j]  # sub-figura corespunzatoare
                )
                axes[i, j].set_title(f"Y = {Y}, theta = {theta}")

    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()
