import numpy as np

def Rastrigin(x):
    num = len(x)
    function = 10*num + np.sum([x0**2 - 10*(np.cos(2*np.pi*x0)) for x0 in x])
    #function = np.sum(np.square(x))
    return function

def coeff_GL(alpha, n):
    h = 1
    vector_coeff = np.zeros((n + 1,), dtype=float)
    vector_coeff[0] = 1  # when omega_0 = 1 for first intera.

    for i in range(len(vector_coeff[1:n]) + 1):
        vector_coeff[i + 1] = (vector_coeff[i] * (1 - ((alpha + 1) / (i + 1))))

    vector_coeff = h**(-alpha)*(vector_coeff)

    return (vector_coeff)



coeff_array_gl = coeff_GL(0.5, 3)
print(coeff_array_gl)

#position_i = np.random.uniform(-5.12, 5.12, [30, 3])

#cost_i = np.array([Rastrigin(particle) for particle in position_i])
#print(position_i)

#for i in range(10):
#    for j in range(3):
#        fitness = np.array([position_i[i] - (1-j)])
#        differential = coeff_array_gl[i] * Rastrigin(fitness)
#        print(fitness)
#print(coeff_array_gl)
#