import numpy as np

def Sphere(x):
    """
    Função de Sphere
    :param x: Vetor de entrada
    :return: Valor da função de Sphere para o vetor de entrada x
    """
    function = np.sum(np.square(x))
    return function

def limits_X(lim_position, boundMax, boundMin):
    for i in range(len(lim_position)):
        if lim_position[i] > boundMax:
            lim_position[i] = boundMax
        if lim_position[i] < boundMin:
            lim_position[i] = boundMin
    return lim_position

position_i = np.random.uniform(-5, 5, (5, 2))
cost_i = np.array([Sphere(particle) for particle in position_i])

pBest = np.copy(position_i)
#print("0.1. ", pBest)
#print('\n 1.', np.shape(pBest))
pBest_cost = np.copy(cost_i)
#print("\n 2.", pBest_cost)

gBest = pBest[np.argmin(pBest_cost)]
#print('\n 3.', gBest)
gBest_cost = np.min(cost_i)
#print('\n 4.', gBest_cost)
max_iter = 10

BestCost = np.zeros(max_iter)

velocity_i = np.zeros((5, 2))


for it in range(max_iter):

    w = 0.9 - 0.4 * (it / max_iter)
    alpha = 0.1 + 1.2 * (it /max_iter)
    beta = 0.1 + 1.2 * (it / max_iter)

    for i in range(5):

        velocity_i[i] = ((w + alpha - 1) * velocity_i[i]
                         + (1) * np.random.rand(2) * (pBest[i] - position_i[i])
                         + (1) * np.random.rand(2) * (gBest - position_i[i])
                         + ((1 / 2) * alpha * (1 - alpha) * velocity_i[i - 1])
                         + ((1 / 6) * alpha * (1 - alpha) * (2 - alpha) * velocity_i[i - 2])

                         + ((1 / 24) * alpha * (1 - alpha) * (2 - alpha) * (3 - alpha) * velocity_i[i - 3]))
        #print('\n iteração:', i, ": ", pBest[i] - position_i[i])
        position_i[i] = ((beta * position_i[i])
                         + velocity_i[i]
                         + ((1/2) * beta * (1-beta) * position_i[i-1])
                         + ((1/6) * beta * (1-beta) * (2-beta) * position_i[i-2])
                         + ((1/24)* beta * (1-beta) * (2-beta) * (3-beta) * position_i[i-3]))
        #position_i[i] = position_i[i] + velocity_i[i]
        position_i[i] = limits_X(position_i[i], 5, -5)


        """Fitness_value = rosenbrock(position_i[i])
        if Fitness_value < pBest_cost[i]:
            pBest[i] = position_i[i]
            pBest_cost[i] = Fitness_value

            if pBest_cost[i] < gBest_cost:
                gBest = pBest[i]
                gBest_cost = pBest_cost[i]"""


    fitness_values = np.array([Sphere(particle) for particle in position_i])
    print(f"\no valor para cost_i: {cost_i} - iteração {it}")
    print(f"o valor para Fitness_value: {fitness_values} - iteração {it}")

    improved_index = np.where(fitness_values < cost_i)
    pBest[improved_index] = position_i[improved_index]

    print(f"o valor para positio_i: {position_i}")
    print(f"o valor para pBest: {pBest}")
    print(f"o valor para pBest com improved_index: {pBest[improved_index]}")

    cost_i[improved_index] = fitness_values[improved_index]

    #print(f"o valor para gBest: {gBest}")
    #print(f"o valor para gBest_cost: {gBest_cost}")

    print(f"- {gBest_cost}")
    print(f"- {np.min(fitness_values)} e {np.argmin(fitness_values)}")

    if np.min(fitness_values) < gBest_cost:

        print(f"- {gBest}")
        print(f"- {position_i[np.argmin(fitness_values)]}")
        gBest = position_i[np.argmin(fitness_values)]
        gBest_cost = np.min(fitness_values)
    BestCost[it] = gBest_cost

print(f"\n: {BestCost}")
