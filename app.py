import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import ga
from ypstruct import structure

# Objective function
def fitness(originalchromosome, individualchromosome):
    return np.mean(np.abs(originalchromosome-individualchromosome))

# Reading original image
original = imageio.imread('chrome.jpg')
directory = os.curdir + '//'
chromlength = 512 * 512 * 3

# Problem Definition
problem = structure()
problem.costfunc = fitness
problem.nvar = chromlength
problem.varmin = 0
problem.varmax = 256
problem.imagearray = original

# Genetic Algorithm parameters
params = structure()
params.maxit = 50
params.npop = 150
params.pc = 50         # Exploration level
params.mu = 1            # Mutation rate
params.sigma = 0.1         # Mutation step
params.beta = 1             # Selection pressure

# Run GA
out = ga.run(problem, params)

# Results
plt.plot(out.bestcost)
#plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel("Iterations")
plt.ylabel("Best Cost")
plt.title("Genetic Algorithm (GA)")
plt.grid(True)
plt.show()

plt.imsave(directory + 'solution2.png', out.result)