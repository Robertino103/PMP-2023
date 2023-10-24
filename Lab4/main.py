import numpy as np
from scipy import stats


def simulate(alpha, target_time):
    num_clients = stats.poisson(mu=lambda_param).rvs()
    client_times = []

    for _ in range(num_clients):
        time_T1 = stats.norm(loc=mean_plasare_comanda, scale=std_dev_plasare_comanda).rvs()
        time_T2 = stats.expon(loc=alpha).rvs()
        total_time = time_T1 + time_T2
        client_times.append(total_time)

    for value in client_times:
        if value > target_time:
            return False
    return True


def get_max_alpha(start_alpha, alpha_step, num_simulations, target_time):
    alpha = start_alpha
    while True:
        print(f"Trying with alpha : {alpha}")
        successful_simulations = 0
        for _ in range(num_simulations):
            if simulate(alpha, target_time):
                successful_simulations += 1

        success_probability = successful_simulations / num_simulations

        if success_probability >= 0.95:
            return alpha

        alpha -= alpha_step


lambda_param = 20
mean_plasare_comanda = 2.0
std_dev_plasare_comanda = 0.5
alpha = 4.0

num_clients = stats.poisson(mu=lambda_param).rvs()

time_T1 = stats.norm(loc=mean_plasare_comanda, scale=std_dev_plasare_comanda).rvs()

time_T2 = stats.expon(loc=alpha).rvs()


print("Numarul de clienti:", num_clients)
print("Timpul pentru plasare si plata:", time_T1)
print("Timpul pentru pregatirea comenzii:", time_T2)

print("Alpha max :", get_max_alpha(start_alpha=20, alpha_step=0.5, num_simulations=20, target_time=15))
