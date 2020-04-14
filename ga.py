import numpy as np
from ypstruct import structure

def imagetochromosome(imagearray, length):
    chromosome = np.reshape(a=imagearray, newshape=length)
    return chromosome

def chromosometoimage(chromosome, imageshape):
    imagearray = np.reshape(a=chromosome, newshape=imageshape)
    return imagearray

def run(problem, params):
    
    # Problem Information
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax
    imagearray = problem.imagearray
    chromlength = problem.chromlength

    # Parameters
    maxit = params.maxit
    npop = params.npop
    pc = params.pc
    nc = int(np.round(pc*npop/2)*2)
    mu = params.mu
    sigma = params.sigma
    beta = params.beta

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.chromosome = None
    empty_individual.cost = None

    # Convert Image Array into Chromosome
    originalchromosome = imagetochromosome(imagearray, nvar) 

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = np.inf

    # Initialize Population
    pop = empty_individual.repeat(npop)

    for i in range(0, npop):
        pop[i].chromosome = np.random.randint(varmax, size=nvar)
        #pop[i].chromosome = np.full(chromlength, varmax-1)
        pop[i].cost = costfunc(originalchromosome, pop[i].chromosome)
        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()

    # Best Cost of Interations
    bestcost = np.empty(maxit)

    # Main Loop
    for it in range(maxit):

        costs = np.array([x.cost for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs/avg_cost
        probs = np.exp(-beta*costs)
        
        popc = []
        for _ in range(nc//2):

            # Random Selection
            # q = np.random.permutation(npop)
            # p1 = pop[q[0]]
            # p2 = pop[q[1]]

            # Roulette_Wheel Selection
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]

            # Perform Crossover
            c1, c2 = crossover(p1, p2)

            # Perform Mutation
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)

            # Apply Bounds
            apply_bound(c1, varmin, varmax)
            apply_bound(c2, varmin, varmax)

            # Evaluate First Offspring
            c1.cost = costfunc(originalchromosome, c1.chromosome)
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()

            # Evaluate Second Offspring
            c2.cost = costfunc(originalchromosome, c2.chromosome)
            if c2.cost < bestsol.cost:
                bestsol = c2.deepcopy()

            #Add Offsprings to popc
            popc.append(c1)
            popc.append(c2)

        #Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[0:npop]

        # Store Best Cost
        bestcost[it] = bestsol.cost

        # Show Iteration Information
        print("Iteration {}: Best Cost = {}".format(it, bestsol.cost))

    # Output
    result = chromosometoimage(bestsol.chromosome, imagearray.shape)
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    out.result = result.astype(np.uint8)
    return out

def crossover(p1, p2):
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    alpha = np.random.randint(2, size=c1.chromosome.shape)
    c1.chromosome = alpha*p1.chromosome + (1-alpha)*p2.chromosome
    c2.chromosome = alpha*p2.chromosome + (1-alpha)*p1.chromosome
    return c1, c2

def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = (np.random.rand(*x.chromosome.shape) <= mu)
    ind = np.argwhere(flag)
    normaldistro = np.rint(sigma*np.random.randn(*ind.shape))
    y.chromosome[ind] += normaldistro.astype(int)
    return y

def apply_bound(x, varmin, varmax):
    x.chromosome = np.maximum(x.chromosome, varmin)
    x.chromosome = np.minimum(x.chromosome, varmax-1)

def roulette_wheel_selection(p):
    cs = np.cumsum(p)
    r = np.random.rand()*sum(p)
    ind = np.argwhere(r <= cs)
    return ind[0][0]