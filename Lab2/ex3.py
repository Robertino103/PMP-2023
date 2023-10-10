import numpy as np
import matplotlib.pyplot as plt
import arviz as az

p_stema = 0.3
nr_aruncari = 10
nr_experimente = 100

rezultate_moneda_masluita = np.random.choice(['s', 'b'], size=(nr_experimente, nr_aruncari), p=[1 - p_stema, p_stema])

rezultate_moneda_nemasluita = np.random.choice(['s', 'b'], size=(nr_experimente, nr_aruncari), p=[0.5, 0.5])

rezultate = np.column_stack((rezultate_moneda_masluita, rezultate_moneda_nemasluita))

nr_ss = np.sum((rezultate[:, :2] == 's').all(axis=1))
nr_sb = np.sum((rezultate[:, 0] == 's') & (rezultate[:, 1] == 'b'))
nr_bs = np.sum((rezultate[:, 0] == 'b') & (rezultate[:, 1] == 's'))
nr_bb = np.sum((rezultate[:, :2] == 'b').all(axis=1))

print(f"Numarul de rezultate 'ss': {nr_ss}")
print(f"Numarul de rezultate 'sb': {nr_sb}")
print(f"Numarul de rezultate 'bs': {nr_bs}")
print(f"Numarul de rezultate 'bb': {nr_bb}")

labels = ['ss', 'sb', 'bs', 'bb']
values = [nr_ss, nr_sb, nr_bs, nr_bb]
plt.pie(values, labels=labels)
plt.xlabel('Rezultate')
plt.ylabel('Numarul de aparitii')
plt.title('Distribuția')
# az.plot_posterior({'ss': nr_ss, 'sb': nr_sb, 'bs': nr_bs, 'bb': nr_bb})
plt.show()
