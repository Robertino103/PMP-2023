import random
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

# Subiectul 1:

def throw_biased_coin():
    # Returneaza 1 cu probabilitatea 1/3 si 0 cu probabilitatea 2/3
    return 1 if random.random() < 1/3 else 0

def throw_fair_coin():
    # 0 sau 1 cu p=1/2
    return random.choice([0, 1])

def sim_game():
    Player1 = ''
    Player2 = ''

    if throw_fair_coin() == 0:
        Player1 = 'P0'
        Player2 = 'P1'
    else:
        Player1 = 'P1'
        Player2 = 'P0'

    # S-a ales ordinea jucatorilor.. se incepe jocul propriu zis:

    # Prima Runda :
    to_turn = Player1

    if to_turn == 'P0':
        # P0 va fi necinstit si va arunca cu moneda masluita
        n = throw_biased_coin()
    elif to_turn == 'P1':
        n = throw_fair_coin()

    # A doua Runda :
    to_turn = Player2

    if to_turn == 'P0':
        m = n+1
        nr_steme = 0
        for i in range(m):
            if (throw_biased_coin() == 1):
                nr_steme += 1

    elif to_turn == 'P1':
        m = n+1
        nr_steme = 0
        for i in range(m):
            if (throw_fair_coin() == 1):
                nr_steme += 1

    winner = None

    if n>=nr_steme:
        # print(Player1 + " a castigat")
        if Player1 == 'P0':
            winner = 0
        if Player1 == 'P1':
            winner = 1
    else:
        # print(Player2 + " a castigat")
        if Player2 == 'P0':
            winner = 0
        if Player2 == 'P1':
            winner = 1

    # Returnam playerul care a inceput, numarul de steme din prima runda, numarul de steme din a doua runda si castigatorul
    return (Player1, n, nr_steme, winner)


def make_bayesian_network():
    model = BayesianNetwork([('P0', 'W'), ('P1', 'W')])
    p0 = TabularCPD(variable="P0", variable_card=2, values=[[0.33], [0.66]])
    p1 = TabularCPD(variable="P1", variable_card=2, values=[[0.5], [0.5]])
    first = TabularCPD(variable="first", variable_card=2, values=[[0.5], [0.5]])
    r1 = TabularCPD(variable='R1', variable_card=2,
                        values=[[0.33, 0.5],
                                [0.66, 0.5]],
                        evidence=['first'],
                        evidence_card=[2])
    r2 = TabularCPD(variable='R2', variable_card=3,
                        values=[[0.5, 0.25, 0.33, 0.12],
                                [0.5, 0.5, 0.66, 0.44],
                                [0, 0.25, 0, 0.44]],
                        evidence=['first', 'R1'],
                        evidence_card=[2, 2])


if __name__ == "__main__":

    # S1.1:
    nr_win_p0 = 0
    nr_win_p1 = 0

    # Simulam jocul de 20k ori si numaram castigurile (returnate pe pozitia 3 in tupla de return a sim_game())
    for i in range(20000):
        if sim_game()[3] == 0:
            nr_win_p0 += 1
        else:
            nr_win_p1 += 1

    print("Probabilitate castig P0 : " + str(nr_win_p0/20000))
    print("Probabilitate castig P1 : " + str(nr_win_p1/20000))

    # S1.2
    model = BayesianNetwork([('FirstPlayer', 'StemeR1'), ('FirstPlayer', 'StemeR2'), ('StemeR1', 'Winner')])
    # FirstPlayer = Primul jucator
    # StemeR1 = Numarul de steme din prima runda = {0,1}
    # StemeR2 = Numarul de steme din a doua runda
    # Winner = Castigatorul jocului

    data = []
    for i in range(20000):
        game_result = sim_game()
        data.append({'FirstPlayer': game_result[0], 'StemeR1': game_result[1], 'StemeR2': game_result[2],
                     'Winner': game_result[3]})

    data = pd.DataFrame(data)

    model.fit(data, estimator=MaximumLikelihoodEstimator)

    # S1.3
    # Determinam fata monedei din prima runda in functie de faptul ca in runda 2 nu s-a obtinut nicio stema
    inference = VariableElimination(model)
    print(inference.query(variables=['StemeR1'], evidence={'StemeR2': 0}))

    # Daca StemeR1(0) are probabilitate mai mare, inseamna ca e mai probabil ca in prima runda sa fi picat 'ban'
    # Similar in mod invers, cu StemeR1(1), unde ar fi mai probabil sa pice 'stema'
