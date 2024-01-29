from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# N = 10000
# x, y = np.random.uniform(-1, 1, size=(2, N))
# inside = (x**2 + y**2) <= 1
# pi = inside.sum()*4/N
# error = abs((pi - np.pi) / pi) * 100
# outside = np.invert(inside)



N = 10000
x = stats.geom.rvs(0.3, size=N)
y = stats.geom.rvs(0.5, size=N)

print(f"x={x}\ny={y}")

inside = x - (y**2) > 0
pi = inside.sum()*4/N
error = abs((pi - np.pi) / pi) * 100
outside = np.invert(inside)
plt.figure(figsize=(8, 8))
plt.plot(x[inside], y[inside], 'b.')
plt.plot(x[outside], y[outside], 'r.')
plt.plot(0, 0, label=f'π*= {pi:4.3f} \n error = {error:4.3f}', alpha=0)
plt.axis('square')
plt.xticks([])
plt.yticks([])
plt.legend(loc=1, frameon=True, framealpha=0.9)
plt.show()