from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# N = 10000
# x = stats.geom.rvs(0.3, size=N)
# y = stats.geom.rvs(0.5, size=N)
#
# print(f"x={x}\ny={y}")
#
# inside = x - (y**2) > 0
# pi = inside.sum()*4/N
# error = abs((pi - np.pi) / pi) * 100
# outside = np.invert(inside)
#
#
# plt.figure(figsize=(8, 8))
# plt.plot(x[inside], y[inside], 'b.')
# plt.plot(x[outside], y[outside], 'r.')
# plt.plot(0, 0, label=f'π*= {pi:4.3f} \n error = {error:4.3f}', alpha=0)
# plt.axis('square')
# plt.xticks([])
# plt.yticks([])
# plt.legend(loc=1, frameon=True, framealpha=0.9)
# plt.show()

def monte_carlo(N=10000):
    theta_X = 0.3
    theta_Y = 0.5
    x = stats.geom.rvs(theta_X, size=N)
    y = stats.geom.rvs(theta_Y, size=N)


    condition = x > y**2
    count = sum(condition)

    predicted = count / N
    return predicted


if __name__ == "__main__":
    # N = 10000
    # for _ in range(N):
    #     x = stats.geom.rvs(0.3, size=1)
    #     y = stats.geom.rvs(0.5, size=1)
    #     inside = x - (y**2) > 0
    #     pi = inside.sum()*4/N
    #
    # predicted_pi = pi
    # error = abs((predicted_pi - np.pi) / predicted_pi) * 100 / N
    #
    # print(error)

    print(monte_carlo(N=10000))
    