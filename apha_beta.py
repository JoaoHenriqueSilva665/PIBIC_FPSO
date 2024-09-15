import numpy as np
import matplotlib.pyplot as plt
from math import gamma

def Sphere(x):
    """
    Função de Sphere
    :param x: Vetor de entrada
    :return: Valor da função de Sphere para o vetor de entrada x

    Domain = [-5.12, 5.12]
    Dimension = d
    """
    function = np.sum(np.square(x))
    return function

def FV_PSO(Particles_dim, alpha_values, beta_values, Function, c1=1, c2=1):
    # velocity_i = np.zeros(Particles_dim)
    velocity_i = np.random.uniform(vMin_bound, vMax_bound, Particles_dim)
    position_i = np.random.uniform(bound_min, bound_max, (Particles_dim))
    cost_i = np.array([Function(particle) for particle in position_i])
    pBest = np.copy(position_i)
    pBest_cost = np.copy(cost_i)
    gBest = pBest[np.argmin(pBest_cost)]
    gBest_cost = np.min(cost_i)
    BestCost = np.zeros(max_iter)

    for it in range(max_iter):

        rand = np.random.rand()
        zr = 4 * rand * (1 - rand)
        w_values = ((0.9 - 0.4) * (max_iter - it) / max_iter) + 0.4 * zr
        #w_values= 1

        for i in range(Particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            velocity_i[i] = (
                    ((w_values) + alpha_values[it] - 1) * velocity_i[i]
                    + c1 * r1 * (pBest[i] - position_i[i])
                    + c2 * r2 * (gBest - position_i[i])
                    + ((1 / 2) * alpha_values[it] * (1 - alpha_values[it]) * velocity_i[i - 1])
                    + ((1 / 6) * alpha_values[it] * (1 - alpha_values[it]) * (
                    2 - alpha_values[it]) * velocity_i[i - 2])
                    + ((1 / 24) * alpha_values[it] * (1 - alpha_values[it]) * (
                    2 - alpha_values[it]) * (3 - alpha_values[it]) * velocity_i[
                           i - 3])
            )
            velocity_i[i] = np.clip(velocity_i[i], vMin_bound, vMax_bound)

            position_i[i] = position_i[i] + velocity_i[i]
            position_i[i] = np.clip(position_i[i], bound_min, bound_max)

        fitness_values = np.array([Function(particle) for particle in position_i])
        improved_index = np.where(fitness_values < pBest_cost)[0]
        pBest[improved_index] = position_i[improved_index]
        pBest_cost[improved_index] = fitness_values[improved_index]
        min_fitness_value = np.min(fitness_values)
        if min_fitness_value < gBest_cost:
            gBest_cost = min_fitness_value
            gBest = position_i[np.argmin(fitness_values)]

        BestCost[it] = gBest_cost

    return gBest, gBest_cost, BestCost
def FP_PSO(Particles_dim, alpha_values, beta_values, Function, c1=1, c2=1):
    # velocity_i = np.zeros(Particles_dim)
    velocity_i = np.random.uniform(vMin_bound, vMax_bound, Particles_dim)
    position_i = np.random.uniform(bound_min, bound_max, (Particles_dim))
    cost_i = np.array([Function(particle) for particle in position_i])
    pBest = np.copy(position_i)
    pBest_cost = np.copy(cost_i)
    gBest = pBest[np.argmin(pBest_cost)]
    gBest_cost = np.min(cost_i)
    BestCost = np.zeros(max_iter)

    for it in range(max_iter):

        rand = np.random.rand()
        zr = 4 * rand * (1 - rand)
        w_values = ((0.9 - 0.4) * (max_iter - it) / max_iter) + 0.4 * zr

        for i in range(Particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            velocity_i[i] = (
                    ((w_values)) * velocity_i[i]
                    + c1 * r1 * (pBest[i] - position_i[i])
                    + c2 * r2 * (gBest - position_i[i]))
            velocity_i[i] = np.clip(velocity_i[i], vMin_bound, vMax_bound)

            position_i[i] = (
                    beta_values[it] * position_i[i]
                    + velocity_i[i]
                    + ((1 / 2) * beta_values[it] * (1 - beta_values[it]) * (position_i[i - 1]))
                    + ((1 / 6) * beta_values[it] * (1 - beta_values[it]) * (2 - beta_values[it]) * (
                position_i[i - 2]))
                    + ((1 / 24) * beta_values[it] * (1 - beta_values[it]) * (
                    2 - beta_values[it]) * (3 - beta_values[it]) * (
                           position_i[i - 3]))
            )
            position_i[i] = np.clip(position_i[i], bound_min, bound_max)

        fitness_values = np.array([Function(particle) for particle in position_i])
        improved_index = np.where(fitness_values < pBest_cost)[0]
        pBest[improved_index] = position_i[improved_index]
        pBest_cost[improved_index] = fitness_values[improved_index]
        min_fitness_value = np.min(fitness_values)
        if min_fitness_value < gBest_cost:
            gBest_cost = min_fitness_value
            gBest = position_i[np.argmin(fitness_values)]

        BestCost[it] = gBest_cost

    return gBest, gBest_cost, BestCost

def FVP_PSO(Particles_dim, alpha_values, beta_values, Function, c1=1, c2=1):
    # velocity_i = np.zeros(Particles_dim)
    velocity_i = np.random.uniform(vMin_bound, vMax_bound, Particles_dim)
    position_i = np.random.uniform(bound_min, bound_max, (Particles_dim))
    cost_i = np.array([Function(particle) for particle in position_i])
    pBest = np.copy(position_i)
    pBest_cost = np.copy(cost_i)
    gBest = pBest[np.argmin(pBest_cost)]
    gBest_cost = np.min(cost_i)
    BestCost = np.zeros(max_iter)

    for it in range(max_iter):

        rand = np.random.rand()
        zr = 4 * rand * (1 - rand)
        w_values = ((0.9 - 0.4) * (max_iter - it) / max_iter) + 0.4 * zr

        for i in range(Particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            velocity_i[i] = (
                    ((w_values) + alpha_values[it] - 1) * velocity_i[i]
                    + c1 * r1 * (pBest[i] - position_i[i])
                    + c2 * r2 * (gBest - position_i[i])
                    + ((1 / 2) * alpha_values[it] * (1 - alpha_values[it]) * velocity_i[i - 1])
                    + ((1 / 6) * alpha_values[it] * (1 - alpha_values[it]) * (
                    2 - alpha_values[it]) * velocity_i[i - 2])
                    + ((1 / 24) * alpha_values[it] * (1 - alpha_values[it]) * (
                    2 - alpha_values[it]) * (3 - alpha_values[it]) * velocity_i[
                           i - 3])
            )
            velocity_i[i] = np.clip(velocity_i[i], vMin_bound, vMax_bound)

            position_i[i] = (
                    beta_values[it] * position_i[i]
                    + velocity_i[i]
                    + ((1 / 2) * beta_values[it] * (1 - beta_values[it]) * (position_i[i - 1]))
                    + ((1 / 6) * beta_values[it] * (1 - beta_values[it]) * (2 - beta_values[it]) * (
                position_i[i - 2]))
                    + ((1 / 24) * beta_values[it] * (1 - beta_values[it]) * (
                    2 - beta_values[it]) * (3 - beta_values[it]) * (
                           position_i[i - 3]))
            )
            position_i[i] = np.clip(position_i[i], bound_min, bound_max)

        fitness_values = np.array([Function(particle) for particle in position_i])
        improved_index = np.where(fitness_values < pBest_cost)[0]
        pBest[improved_index] = position_i[improved_index]
        pBest_cost[improved_index] = fitness_values[improved_index]
        min_fitness_value = np.min(fitness_values)
        if min_fitness_value < gBest_cost:
            gBest_cost = min_fitness_value
            gBest = position_i[np.argmin(fitness_values)]

        BestCost[it] = gBest_cost

    return gBest, gBest_cost, BestCost

Particles = 1000
dim = 10
Particles_dim = (Particles, dim)
max_iter = 300
alpha_values = 0.1 + 1.2 * (np.arange(max_iter) / max_iter)
#alpha_values = np.linspace(0.4, 1.0, num=max_iter)
beta_values = 0.1 + 1.2 * (np.arange(max_iter) / max_iter)
#beta_values = np.linspace(0.452, 1.052, num=300)

rand = np.random.rand()
zr = 4 * rand * (1 - rand)
w_values = ((0.9 - 0.4) * (max_iter - np.arange(max_iter)) / max_iter) + 0.4 * zr

bound_min = (-10)
bound_max = (10)
vMin_bound = (-0.1 * (bound_max - bound_min))
vMax_bound = (0.1 * (bound_max - bound_min))



gBest, gBest_cost, BestCost = FV_PSO(Particles_dim, alpha_values, beta_values, Sphere)
gBest_var_01, gBest_cost_var_01, BestCost_var_01 = FP_PSO(Particles_dim, alpha_values, beta_values,
                                                    Sphere)
gBest_var02, gBest_cost_var02, BestCost_var02 = FVP_PSO(Particles_dim, alpha_values, beta_values,
                                                    Sphere)

plt.rcParams['legend.fontsize'] = 14
plt.plot(BestCost, color="k", linestyle="solid", label="FV_PSO",
             linewidth=1.5)
plt.plot(BestCost_var_01, color="k", linestyle=":", label="FP_PSO",
             linewidth=1.5)
plt.plot(BestCost_var02, color="k", linestyle="-.", label="FVP_PSO",
             linewidth=1.5)

plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')


plt.xlabel("Número de Iteração", fontsize=12)
plt.xlim(0, 300)

plt.ylabel("Valor da Função Objetivo", fontsize=12)
plt.yscale("log")

plt.legend(framealpha=1, frameon=True)
plt.tight_layout()

plt.show()
#de 0.452 ate 1.052