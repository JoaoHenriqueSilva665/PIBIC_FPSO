import matplotlib.pyplot as plt
import numpy as np


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


def Rosenbrock(x):
    """
        Função de Rosenbrock
        :param x: Vetor de entrada
        :return: Valor da função de Rosenbrock para o vetor de entrada x

        Domain = [-5, 5] ou [-10, 10]
        Dimension = d
        """
    n = len(x)
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(n - 1))


def Pass_velocity(vMin_bound, bound_min, vMax_bound, bound_max, Particles_dim):
    all_iterations_velocity = []
    all_iterations_position = []

    for _ in range(3):
        # Gerando a matriz de velocidades para esta iteração
        velocity_i = np.random.uniform(vMin_bound, vMax_bound, (Particles_dim))
        position_i = np.random.uniform(bound_min, bound_max, (Particles_dim))

        # Armazenando a matriz de velocidades atual
        all_iterations_velocity.append(velocity_i)
        all_iterations_position.append(position_i)

    # Convertendo a lista para um array NumPy tridimensional
    all_iterations_velocity = np.array(all_iterations_velocity)
    all_iterations_position = np.array(all_iterations_position)
    return all_iterations_velocity, all_iterations_position


def FPSO(Particles_dim, alpha_values, beta_values, Function, c1=1, c2=1):
    # velocity_i = np.zeros(Particles_dim)
    velocity_i = np.random.uniform(vMin_bound, vMax_bound, Particles_dim)
    position_i = np.random.uniform(bound_min, bound_max, (Particles_dim))
    cost_i = np.array([Function(particle) for particle in position_i])
    pBest = np.copy(position_i)
    pBest_cost = np.copy(cost_i)
    gBest = pBest[np.argmin(pBest_cost)]
    gBest_cost = np.min(cost_i)
    BestCost = np.zeros(max_iter)

    all_iterations_velocity, all_iterations_position = Pass_velocity(vMin_bound, bound_min, vMax_bound, bound_max, Particles_dim)

    trajetoria_velocidade = np.zeros((max_iter, 1))
    trajetoria_posicao = np.zeros((max_iter, 1))
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
                    + ((1 / 2) * alpha_values[it] * (1 - alpha_values[it]) * all_iterations_velocity[0, i])
                    + ((1 / 6) * alpha_values[it] * (1 - alpha_values[it]) * (2 - alpha_values[it]) * all_iterations_velocity[1, i])
                    + ((1 / 24) * alpha_values[it] * (1 - alpha_values[it]) * (2 - alpha_values[it]) * (3 - alpha_values[it]) *
                       all_iterations_velocity[2, i]))
            velocity_i[i] = np.clip(velocity_i[i], vMin_bound, vMax_bound)
            trajetoria_velocidade[it] = velocity_i[i]

            position_i[i] = (
                    beta_values[it] * position_i[i]
                    + velocity_i[i]
                    + ((1 / 2) * beta_values[it] * (1 - beta_values[it]) * (all_iterations_position[0, i]))
                    + ((1 / 6) * beta_values[it] * (1 - beta_values[it]) * (2 - beta_values[it]) * (all_iterations_position[1, i]))
                    + ((1 / 24) * beta_values[it] * (1 - beta_values[it]) * (2 - beta_values[it]) * (3 - beta_values[it]) * (
                all_iterations_position[2, i])))
            position_i[i] = np.clip(position_i[i], bound_min, bound_max)
            trajetoria_posicao[it] = position_i[i]
            # print(position_i)

        all_iterations_velocity = np.roll(all_iterations_velocity, shift=1, axis=0)
        all_iterations_velocity[0] = velocity_i

        all_iterations_position = np.roll(all_iterations_position, shift=1, axis=0)
        all_iterations_position[0] = position_i

        fitness_values = np.array([Function(particle) for particle in position_i])
        improved_index = np.where(fitness_values < pBest_cost)[0]
        pBest[improved_index] = position_i[improved_index]
        pBest_cost[improved_index] = fitness_values[improved_index]
        min_fitness_value = np.min(fitness_values)
        if min_fitness_value < gBest_cost:
            gBest_cost = min_fitness_value
            gBest = position_i[np.argmin(fitness_values)]

        BestCost[it] = gBest_cost

    return gBest, gBest_cost, BestCost, trajetoria_velocidade, trajetoria_posicao


def PSO(Particles_dim, Function, c1=1, c2=1):
    # velocity_i = np.zeros(Particles_dim)
    velocity_i = np.random.uniform(vMin_bound, vMax_bound, Particles_dim)
    position_i = np.random.uniform(bound_min, bound_max, (Particles_dim))
    cost_i = np.array([Function(particle) for particle in position_i])
    pBest = np.copy(position_i)
    pBest_cost = np.copy(cost_i)
    gBest = pBest[np.argmin(pBest_cost)]
    gBest_cost = np.min(cost_i)
    BestCost = np.zeros(max_iter)

    all_iterations_velocity, all_iterations_position = Pass_velocity(vMin_bound, bound_min, vMax_bound, bound_max, Particles_dim)

    trajetoria_velocidade = np.zeros((max_iter, 1))
    trajetoria_posicao = np.zeros((max_iter, 1))
    for it in range(max_iter):

        w = 1
        for i in range(Particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocity_i[i] = (
                    ((w)) * velocity_i[i]
                    + c1 * r1 * (pBest[i] - position_i[i])
                    + c2 * r2 * (gBest - position_i[i]))
            velocity_i[i] = np.clip(velocity_i[i], vMin_bound, vMax_bound)
            trajetoria_velocidade[it] = velocity_i[i]

            position_i[i] = (position_i[i] + velocity_i[i])
            position_i[i] = np.clip(position_i[i], bound_min, bound_max)
            trajetoria_posicao[it] = velocity_i[i]
            # print(position_i)

        all_iterations_velocity = np.roll(all_iterations_velocity, shift=1, axis=0)
        all_iterations_velocity[0] = velocity_i

        all_iterations_position = np.roll(all_iterations_position, shift=1, axis=0)
        all_iterations_position[0] = position_i

        fitness_values = np.array([Function(particle) for particle in position_i])
        improved_index = np.where(fitness_values < pBest_cost)[0]
        pBest[improved_index] = position_i[improved_index]
        pBest_cost[improved_index] = fitness_values[improved_index]
        min_fitness_value = np.min(fitness_values)
        if min_fitness_value < gBest_cost:
            gBest_cost = min_fitness_value
            gBest = position_i[np.argmin(fitness_values)]
        BestCost[it] = gBest_cost
    return gBest, gBest_cost, BestCost, trajetoria_velocidade, trajetoria_posicao


Particles = 1
dim = 1
Particles_dim = (Particles, dim)
max_iter = 50
alpha_values = 0.1 + (1.2 * (np.arange(max_iter) / max_iter))
beta_values = 0.1 + (1.2 * (np.arange(max_iter) / max_iter))
iter = 3
bound_min = (-100)
bound_max = (100)
vMin_bound = (-0.1 * (bound_max - bound_min))
vMax_bound = (0.1 * (bound_max - bound_min))

gBest, gBest_cost, BestCost, trajetoria_velocidade, trajetoria_posicao = FPSO(Particles_dim, alpha_values, beta_values, Rosenbrock)
gBest_classic, gBest_cost_classic, BestCost_classic, trajetoria_velocidade_classic, trajetoria_posicao_classica = PSO(Particles_dim, Rosenbrock)
all_iter, all_iter_position = Pass_velocity(vMin_bound, bound_min, vMax_bound, bound_max, Particles_dim)

# Plotar Gráficos
plt.plot(range(max_iter), trajetoria_posicao_classica, marker='.',linestyle="dotted", color="k", label="PSO", linewidth=1.5)
plt.plot(range(max_iter), trajetoria_posicao, marker='.', linestyle="solid", color="k", label="FPSO", linewidth=1.5)

plt.xlabel('Iteração', fontsize=12)
plt.ylabel('Posição da Partícula', fontsize=12)

plt.xlim(0, max_iter)
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')

# plt.grid(True, which='both')
plt.legend(framealpha=1, frameon=True)
plt.tight_layout()
plt.savefig("Imagens/Trajetoria_posicao_01_.svg", format="svg")
plt.show()
