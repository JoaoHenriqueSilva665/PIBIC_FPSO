"""
Bibliotecas importantes para o projeto
"""
import numpy as np
import matplotlib.pyplot as plt

def DesignOfAGearTrain(x):
    x1, x2, x3, x4 = x
    return ((1 / 6.931) - (x3 * x2) / (x1 * x4))**2

def FPSO(Particles_dim, Function):
    velocity_i = np.zeros(Particles_dim)
    # velocity_i = np.random.uniform([d_min, D_min, N_min], [d_max, D_max, N_max], Particles_dim)
    position_i = np.random.uniform(bound_min, bound_max, (Particles_dim))
    # print(f"vetor de particulas: \n{position_i} \n")
    cost_i = np.array([Function(particle) for particle in position_i])
    # print(f"resultado na função objetivo: \n{cost_i} \n")

    pBest = np.copy(position_i)  # Copia de Position
    pBest_cost = np.copy(cost_i)  # Copia de Cost

    gBest = pBest[np.argmin(pBest_cost)]
    # print(f"A posição responsável pelo minimo: \n{gBest} \n")
    gBest_cost = np.min(cost_i)
    # print(f"O minimo: \n{gBest_cost} \n")

    BestCost = np.zeros(max_iter)

    for it in range(max_iter):

        rand = np.random.rand()
        zr = 4 * rand * (1 - rand)
        w = ((0.9 - 0.4) * (max_iter - it) / max_iter) * 0.4 * zr

        alpha = 0.1 + (1.2 * (it / max_iter))
        beta = 0.1 + (1.2 * (it / max_iter))
        c1, c2 = 1, 1
        # count = 0
        # print(f"Alguna valores importante \n"
        #      f"1.{w}\n "
        #      f"2.{alpha}\n "
        #      f"3.{beta}\n "
        #      f"4.{c1} e {c2}\n")
        for i in range(Particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            # print(f"(ANTES VELOCITY) em i={i}:\n {velocity_i} ")
            velocity_i[i] = (
                    (w + alpha - 1) * velocity_i[i]
                    + c1 * r1 * (pBest[i] - position_i[i])
                    + c2 * r2 * (gBest - position_i[i])
                    + ((1 / 2) * alpha * (1 - alpha) * velocity_i[i - 1])
                    + ((1 / 6) * alpha * (1 - alpha) * (2 - alpha) * velocity_i[i - 2])
                    + ((1 / 24) * alpha * (1 - alpha) * (2 - alpha) * (3 - alpha) * velocity_i[
                i - 3])
            )
            # print(f"(DEPOIS VELOCITY) em i={i}:\n {velocity_i}\n")
            # print("##### verificação de velocidade #####")
            velocity_i[i] = np.clip(velocity_i[i], vMin_bound, vMax_bound)
            # print(f"###### Depois da verificação de velocidade:\n {velocity_i}\n")

            # print(f"(ANTES POSITION) em i={i}:\n {position_i}")
            position_i[i] = (
                    beta * position_i[i]
                    + velocity_i[i]
                    + ((1 / 2) * beta * (1 - beta) * (position_i[i - 1]))
                    + ((1 / 6) * beta * (1 - beta) * (2 - beta) * (position_i[i - 2]))
                    + ((1 / 24) * beta * (1 - beta) * (2 - beta) * (3 - beta) * (
                position_i[i - 3]))
            )

            # print("##### verificação de Posição #####")
            position_i[i] = np.clip(position_i[i], bound_min, bound_max)

            # print(f"###### Depois da verificação:\n {position_i}\n")

        # Armazenamento do melhor custo encontrado na iteração atual
        # BestCost[it] = gBest_cost

        # Cálculo dos valores de fitness para todas as partículas na posição atual

        fitness_values = np.array([Function(particle) for particle in position_i])
        # print(f"O resultado da função objetivo!!!: {fitness_values}")

        # Verificação das partículas que melhoraram sua posição
        improved_index = np.where(fitness_values < pBest_cost)
        # print(f"Onde a condição de pbest > fitness_values:\n {improved_index}\n")

        # print(f"comparação:\n antes -> \n{pBest}\n")
        # Atualização de pBest para partículas que encontraram uma melhor solução
        pBest[improved_index] = position_i[improved_index]
        # print(f"comparação:\n depois -> \n{pBest}\n ")

        # print(f"comparação:\n antes -> \n{pBest_cost}\n")
        pBest_cost[improved_index] = fitness_values[improved_index]
        # print(f"comparação:\n depois -> \n{pBest_cost}\n ")

        # Verificação se a melhor solução global foi encontrada
        min_fitness_value = np.min(fitness_values)
        # print(f"O minimo!: {min_fitness_value}")

        # print(f"O otimo global até agora.. {it}: \n{gBest_cost}\n")
        # print(f"A melhor particula global até agora..: \n{gBest}\n")
        # print(f"A melhor particula local até agora..: \n{pBest}\n")
        # print(f"array de posições até agora..: \n{position_i}\n")
        if min_fitness_value < gBest_cost:
            gBest_cost = min_fitness_value
            gBest = position_i[np.argmin(fitness_values)]

        # Armazenamento do melhor custo encontrado na iteração atual
        BestCost[it] = gBest_cost

    #gBest = np.round(gBest)
    #gBest_cost = np.round(gBest_cost)
    #BestCost = np.round(BestCost)

    return gBest, gBest_cost, BestCost


def PSO(Particles_dim, Function):
    velocity_i = np.zeros(Particles_dim)
    # velocity_i = np.random.uniform([d_min, D_min, N_min], [d_max, D_max, N_max], Particles_dim)
    position_i = np.random.uniform((bound_min), (bound_max), (Particles_dim))
    # print(f"vetor de particulas: \n{position_i} \n")

    cost_i = np.array([Function(particle) for particle in position_i])
    # print(f"resultado na função objetivo: \n{cost_i} \n")

    pBest = np.copy(position_i)  # Copia de Position
    pBest_cost = np.copy(cost_i)  # Copia de Cost

    gBest = pBest[np.argmin(pBest_cost)]
    # print(f"A posição responsável pelo minimo: \n{gBest} \n")
    gBest_cost = np.min(cost_i)
    # print(f"O minimo: \n{gBest_cost} \n")

    BestCost = np.zeros(max_iter)

    for it in range(max_iter):

        rand = np.random.rand()
        zr = 4 * rand * (1 - rand)
        w = ((0.9 - 0.4) * (max_iter - it) / max_iter) * 0.4 * zr

        alpha = 1
        beta = 1
        c1, c2 = 1, 1
        # count = 0
        # print(f"Alguna valores importante \n"
        #      f"1.{w}\n "
        #      f"2.{alpha}\n "
        #      f"3.{beta}\n "
        #      f"4.{c1} e {c2}\n")
        for i in range(Particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            # print(f"(ANTES VELOCITY) em i={i}:\n {velocity_i} ")
            velocity_i[i] = (
                    (w + alpha - 1) * velocity_i[i]
                    + c1 * r1 * (pBest[i] - position_i[i])
                    + c2 * r2 * (gBest - position_i[i])
                    + ((1 / 2) * alpha * (1 - alpha) * velocity_i[i - 1])
                    + ((1 / 6) * alpha * (1 - alpha) * (2 - alpha) * velocity_i[i - 2])
                    + ((1 / 24) * alpha * (1 - alpha) * (2 - alpha) * (3 - alpha) * velocity_i[
                i - 3])
            )
            # print(f"(DEPOIS VELOCITY) em i={i}:\n {velocity_i}\n")
            # print("##### verificação de velocidade #####")
            velocity_i[i] = np.clip(velocity_i[i], vMin_bound, vMax_bound)
            # print(f"###### Depois da verificação de velocidade:\n {velocity_i}\n")

            # print(f"(ANTES POSITION) em i={i}:\n {position_i}")
            position_i[i] = (
                    beta * position_i[i]
                    + velocity_i[i]
                    + ((1 / 2) * beta * (1 - beta) * (position_i[i - 1]))
                    + ((1 / 6) * beta * (1 - beta) * (2 - beta) * (position_i[i - 2]))
                    + ((1 / 24) * beta * (1 - beta) * (2 - beta) * (3 - beta) * (
                position_i[i - 3]))
            )

            # print(f"##### Depois da validação: \n{position_i}\n #####")
            # print("##### verificação de Posição #####")
            position_i[i] = np.clip(position_i[i], bound_min, bound_max)

            # print(f"###### Depois da verificação:\n {position_i}\n")

        # Armazenamento do melhor custo encontrado na iteração atual
        # BestCost[it] = gBest_cost

        # Cálculo dos valores de fitness para todas as partículas na posição atual

        fitness_values = np.array([Function(particle) for particle in position_i])
        # print(f"O resultado da função objetivo!!!: {fitness_values}")

        # Verificação das partículas que melhoraram sua posição
        improved_index = np.where(fitness_values < pBest_cost)
        # print(f"Onde a condição de pbest > fitness_values:\n {improved_index}\n")

        # print(f"comparação:\n antes -> \n{pBest}\n")
        # Atualização de pBest para partículas que encontraram uma melhor solução
        pBest[improved_index] = position_i[improved_index]
        # print(f"comparação:\n depois -> \n{pBest}\n ")

        # print(f"comparação:\n antes -> \n{pBest_cost}\n")
        pBest_cost[improved_index] = fitness_values[improved_index]
        # print(f"comparação:\n depois -> \n{pBest_cost}\n ")

        # Verificação se a melhor solução global foi encontrada
        min_fitness_value = np.min(fitness_values)
        # print(f"O minimo!: {min_fitness_value}")

        # print(f"O otimo global até agora.. {it}: \n{gBest_cost}\n")
        # print(f"A melhor particula global até agora..: \n{gBest}\n")
        # print(f"A melhor particula local até agora..: \n{pBest}\n")
        # print(f"array de posições até agora..: \n{position_i}\n")
        if min_fitness_value < gBest_cost:
            gBest_cost = min_fitness_value
            gBest = position_i[np.argmin(fitness_values)]

        # Armazenamento do melhor custo encontrado na iteração atual
        BestCost[it] = gBest_cost

    return gBest, gBest_cost, BestCost


def Test_Cycle(initial_vector_pBest, initial_vector_pBest_classic, BestCost, BestCost_classic,
               gBest_cost, gBest_cost_classic, Particles_dim, Function, iter=100):
    Function = Function

    # FPSO
    size_vector_gBest = len(initial_vector_pBest)
    all_iterations_gBest = np.zeros((iter, size_vector_gBest))
    all_interations_BestCost = np.zeros((iter, len(BestCost)))
    # gBest_cost_values = np.zeros(iter)
    # print(gBest_cost_values, "\n")

    # PSO
    size_vector_gBest_classic = len(initial_vector_pBest_classic)
    all_iterations_gBest_classic = np.zeros((iter, size_vector_gBest_classic))
    all_interations_BestCost_classic = np.zeros((iter, len(BestCost_classic)))

    # PSO_gBest, PSO_gBest_cost, PSO_BestCost = FPSO(Particles_dim, Function)
    # PSO_gBest_classic, PSO_gBest_cost_classic, PSO_BestCost_classic = PSO_classic(
    #    Particles_dim,
    #   Function)

    all_iterations_gBest[0] = initial_vector_pBest
    all_interations_BestCost[0] = BestCost
    # gBest_cost_values[0] = gBest_cost
    # print(gBest_cost_values, "\n")

    all_iterations_gBest_classic[0] = initial_vector_pBest_classic
    all_interations_BestCost_classic[0] = BestCost_classic

    # print(initial_vector_pBest)

    for i in range(1, iter):
        new_vector_pBest = gBest
        new_vector_BestCost = BestCost
        # new_vector_gBest_cost = PSO_gBest_cost
        # print(new_vector_gBest_cost, "\n")

        new_vector_pBest_classic = gBest_classic
        new_vector_BestCost_classic = BestCost_classic

        all_iterations_gBest[i] = new_vector_pBest
        all_interations_BestCost[i] = new_vector_BestCost
        # gBest_cost_values[i] = new_vector_gBest_cost
        # print(gBest_cost_values, "\n")

        all_iterations_gBest_classic[i] = new_vector_pBest_classic
        all_interations_BestCost_classic[i] = new_vector_BestCost_classic

        # Calculando a média de cada coordenada
    average_vector_gBest = np.mean(all_iterations_gBest, axis=0)
    average_vector_BestCost = np.mean(all_interations_BestCost, axis=0)

    average_vector_gBest_classic = np.mean(all_iterations_gBest_classic, axis=0)
    average_vector_BestCost_classic = np.mean(all_interations_BestCost_classic, axis=0)

    value_costum_function = Function(average_vector_gBest)
    value_costum_function_classic = Function(average_vector_gBest_classic)

    print(f'BEST FINTNESS  VALUE (FPSO) = {value_costum_function}')
    print(f'BEST FINTNESS  VALUE da iteração -> {i} (FPSO)= {gBest_cost}')
    print(f"Média das Cordenadas (FPSO): {np.round(average_vector_gBest)}\n")

    print(f'BEST FINTNESS  VALUE (PSO) = {value_costum_function_classic}')
    print(f'BEST FINTNESS  VALUE da iteração -> {i} (PSO)= {gBest_cost_classic}')
    print(f"Cordenadas (PSO): {np.round(average_vector_gBest_classic)}\n")

    plt.rcParams['legend.fontsize'] = 12
    plt.plot(average_vector_BestCost_classic, color="k", linestyle="solid", label="PSO",
             linewidth=1)
    plt.plot(average_vector_BestCost, color="k", linestyle="dashdot", label="FPSO",
             linewidth=1)

    plt.xlabel("$t$")
    plt.xlim(0, max_iter)

    plt.ylabel("$f_2$")
    #plt.yscale("log")

    plt.legend(framealpha=1, frameon=True)
    plt.tight_layout()
    # plt.savefig('plot.png')
    plt.show()

    return (average_vector_gBest,
            average_vector_gBest_classic,
            average_vector_BestCost,
            average_vector_BestCost_classic,
            all_iterations_gBest,
            all_iterations_gBest_classic)


"""
Condições de entrada para a função PSO e Função teste
"""
Particles = 400
dim = 4
Particles_dim = (Particles, dim)
max_iter = 100

bound_min = [12, 12, 12, 12]
bound_max = [60, 60, 60, 60]

vMin_bound = [-0.1 * (bound_max[0] - bound_min[0]),
              -0.1 * (bound_max[1] - bound_min[1]),
              -0.1 * (bound_max[2] - bound_min[2]),
              -0.1 * (bound_max[3] - bound_min[3])]

vMax_bound = [0.1 * (bound_max[0] - bound_min[0]),
              0.1 * (bound_max[1] - bound_min[1]),
              0.1 * (bound_max[2] - bound_min[2]),
              0.1 * (bound_max[3] - bound_min[3])]

gBest, gBest_cost, BestCost = FPSO(Particles_dim, DesignOfAGearTrain)
gBest_classic, gBest_cost_classic, BestCost_classic = PSO(Particles_dim,
                                                          DesignOfAGearTrain)

(average_vector_pBest,
 average_vector_pBest_classic,
 average_vector_BestCost,
 average_vector_BestCost_classic,
 all_iterations, all_iterations_classic) = Test_Cycle(gBest, gBest_classic, BestCost,
                                                      BestCost_classic, gBest_cost,
                                                      gBest_cost_classic, Particles_dim,
                                                      DesignOfAGearTrain)
