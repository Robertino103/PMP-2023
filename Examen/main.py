import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az


def read_data(filename):
    csv_path = filename

    df = pd.read_csv(csv_path)

    # Preprocesare valori lipsa:
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df = df.dropna(subset=['Embarked'])

    return df


if __name__ == "__main__":

    df = read_data(filename='Titanic.csv')

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta_pclass = pm.Normal('beta_pclass', mu=0, sigma=10)
        beta_age = pm.Normal('beta_age', mu=0, sigma=10)

        # Model liniar:
        mu = alpha + beta_pclass * df['Pclass'] + beta_age * df['Age']

        # Definirea distributiei pentru observed (variabila Survived)
        medv = pm.Normal('Survived', mu=mu, sigma=1, observed=df['Survived'])

        # Estimarea parametrilor:
        trace = pm.sample(1000, tune=500, chains=2)

    az.plot_posterior(trace)

    # Calculam HDI pentru a observa care din variabile influenteaza mai mult rezultatul:
    summary = az.summary(trace, hdi_prob=0.95)
    print(summary)
