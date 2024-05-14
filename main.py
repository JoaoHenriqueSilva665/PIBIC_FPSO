"""
Bibliotecas importantes para o projeto
"""
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

"""
Funções Teste
"""
def Rastrigin(x):
    """
    Função de Rastrigin
    :param x: Vetor de entrada
    :return: Valor da função de Rastringin para o vetor de entrada x
    """
    num = len(x)
    function = 10*num + np.sum([x0**2 - 10*(np.cos(2*np.pi*x0)) for x0 in x])
    return function

def Sphere(x):
    """
    Função de Sphere
    :param x: Vetor de entrada
    :return: Valor da função de Sphere para o vetor de entrada x
    """
    function = np.sum(np.square(x))
    return function

def rosenbrock(x):
    """
    Função de Rosenbrock
    :param x: Vetor de entrada
    :return: Valor da função de Rosenbrock para o vetor de entrada x
    """
    sum = 0
    for i in range(len(x) - 1):
        sum += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
    return sum

"""
Função para restrigir o espaço viável
"""
def limits_X(lim_position, boundMax, boundMin):
    for i in range(len(lim_position)):
        if lim_position[i] > boundMax:
            lim_position[i] = boundMax
        if lim_position[i] < boundMin:
            lim_position[i] = boundMin
    return lim_position

"""
Condições de entrada para a função PSO e Função teste
"""
Particles = 30
dim = 10
Particles_dim = (Particles, dim)
max_iter = 300
c1, c2 = 1, 1
bMin, bMax = -30, 30

"""
Vetores iniciais
"""
velocity_i = np.zeros(Particles_dim)
position_i = np.random.uniform(bMin, bMax, Particles_dim)
cost_i = np.array([rosenbrock(particle) for particle in position_i])

"""
Cópias do vetor ~position_i e ~cost_i e Possíveis modificações
"""
pBest = np.copy(position_i)
pBest_cost = np.copy(cost_i)

"""
VETOR associado ao menor valor para a funçao objetivo f(x)
"""
gBest = pBest[np.argmin(pBest_cost)]

"""
O menor VALOR para a função objetivo f(x)
"""
gBest_cost = np.min(cost_i)

"""
Vetor que vai ser armazenado os melhor valores globais
"""
BestCost = np.zeros(max_iter)


for it in range(max_iter):

    """
    Critério de Convergência para o projeto
    """
    w = 0.9 - 0.6 * (it / max_iter)
    alpha = 0.1 + 1.2 * (it /max_iter)
    beta = 0.1 + 1.2 * (it / max_iter)

    for i in range(Particles):

        """
        Função de velocidade para o enxame de partículas
        """
        velocity_i[i] = ((w + alpha - 1) * velocity_i[i]
                         + (c1) * np.random.rand(dim) * (pBest[i] - position_i[i])
                         + (c2) * np.random.rand(dim) * (gBest - position_i[i])
                         + ((1 / 2) * alpha * (1 - alpha) * velocity_i[i - 1])
                         + ((1 / 6) * alpha * (1 - alpha) * (2 - alpha) * velocity_i[i - 2])
                         + ((1 / 24) * alpha * (1 - alpha) * (2 - alpha) * (3 - alpha) * velocity_i[i - 3]))

        """
        Função Posição para o enxame
        """
        position_i[i] = ((beta * position_i[i])
                         + velocity_i[i]
                         + ((1/2) * beta * (1-beta) * position_i[i-1])
                         + ((1/6) * beta * (1-beta) * (2-beta) * position_i[i-2])
                         + ((1/24)* beta * (1-beta) * (2-beta) * (3-beta) * position_i[i-3]))
        #position_i[i] = position_i[i] + velocity_i[i]
        position_i[i] = limits_X(position_i[i], bMax, bMin)


        """
        Validação simples Nível = 0
        """
        """Fitness_value = rosenbrock(position_i[i])
        if Fitness_value < pBest_cost[i]:
            pBest[i] = position_i[i]
            pBest_cost[i] = Fitness_value

            if pBest_cost[i] < gBest_cost:
                gBest = pBest[i]
                gBest_cost = pBest_cost[i]"""

    """
    Validação Robusta Nível = +1
    """
    fitness_values = np.array([rosenbrock(particle) for particle in position_i])

    improved_index = np.where(fitness_values < cost_i)
    pBest[improved_index] = position_i[improved_index]
    cost_i[improved_index] = fitness_values[improved_index]
    if np.min(fitness_values) < gBest_cost:
        gBest = position_i[np.argmin(fitness_values)]
        gBest_cost = np.min(fitness_values)
    BestCost[it] = gBest_cost


"""
Saída Principal
"""
print(f'BEST FINTNESS  VALUE = {gBest_cost}')
print(f"Cordenadas: {np.ceil(gBest)}")

"""
Plot do gráfico de convergência
"""
plt.plot(BestCost)
plt.title('Convergência do PSO para a função de Rastrigin')
plt.xlabel('Iteração')
plt.ylabel('Valor da Função Objetivo')
plt.show()


