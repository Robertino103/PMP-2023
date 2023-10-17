from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx


"""
Un depozit de marfă este echipat cu alarmă de incendiu. 
Pentru o zi dată, probabilitatea să aibă loc un
cutremur in zona depozitului este de 0.05%.
Şansa ca un incendiu să se declanşeze la depozit 
este aproximată la 1% in mod normal dar, dacă a avut loc
un cutremur, aceasta creşte la 3%.
Alarma de incendiu are probabilitatea să se declanşeze
accidental de 0.01%, dar în caz de cutremur această
declanşare accidentală urcă la 2%; 
în caz de incendiu alarma are 95% şansă de declanşare, 
iar dacă a avut loc şi cutremur şi incendiu, 
alarma se declanşează cu probabilitatea de 98%.
2. Ştiind că alarma de incendiu a fost declanşată, 
calculaţi probabilitatea să fi avut loc un cutremur. (1p)
3. Afişaţi probabilitatea ca un incendiu sa fi avut loc, 
fără ca alarma de incendiu să se activeze. (1p)
"""

model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Incendiu', 'Alarma'), ('Cutremur', 'Alarma')])

cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2,
                          values=[[0.9995], [0.0005]])

cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2,
                          values=[[0.99, 0.97],
                                  [0.01, 0.03]],
                          evidence=['Cutremur'], evidence_card=[2])

cpd_alarma = TabularCPD(variable='Alarma', variable_card=2,
                        values=[[0.9999, 0.98, 0.05, 0.02],
                                [0.0001, 0.02, 0.95, 0.98]],
                        evidence=['Cutremur', 'Incendiu'], evidence_card=[2, 2])

model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarma)

assert model.check_model()

infer = VariableElimination(model)

prob_ex2 = infer.query(variables=['Cutremur'], evidence={'Alarma': 1})
print(prob_ex2)

prob_ex3 = infer.query(variables=['Incendiu'], evidence={'Alarma': 0})
print(prob_ex3)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()
