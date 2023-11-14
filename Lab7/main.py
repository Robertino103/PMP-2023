import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np

df = pd.read_csv('auto-mpg.csv')

df = df.dropna(subset=['horsepower', 'mpg'])
df = df[df['horsepower'] != '?']
df = df[df['mpg'] != '?']

df['horsepower'] = pd.to_numeric(df['horsepower'])
df['mpg'] = pd.to_numeric(df['mpg'])

plt.scatter(df['horsepower'], df['mpg'])
plt.xlabel('horsepower')
plt.ylabel('mpg')
plt.show()
