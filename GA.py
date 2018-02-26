import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

""" Simple genetic algorithm to maximize functions """


class GA:

    def __init__(self, func, domain, range, populationSize, iterations, mutationRate):
        """
        :param func: function to maximize
        :param domain: [min, max] x values
        :param range:  [min, max] y values
        :param populationSize: # of points searching
        :param iterations: # of generations
        :param mutationRate: # 0 <= mutation rate <= 1
        """

        self.func = func
        self.domain = domain
        self.range = range
        self.populationSize = populationSize
        self.iterations = iterations
        self.mutationRate = mutationRate

        self.population = []
        self.populationFitness = 0

        self.best = None
        self.worst = None
        self.average = None

        self.popHistory = []


    def initializePopulation(self):
        """ Create an initial population of points """

        for i in range(self.populationSize):
            x = np.random.uniform(self.domain[0], self.domain[1])
            y = np.random.uniform(self.range[0], self.range[1])
            fit = self.func([x, y])

            self.population.append(Point(x, y, fit))

        self.popHistory.append(self.population)
        self.popFitness()


    def debug(self):
        """ Used for debugging - prints x and y values for each member of the population """
        count = 1
        for a in self.population:
            print(count,'#  ', a.x, a.y)
            count += 1


    def evaluate(self):
        """ Update populations best, worst, and average """
        self.popFitness()

        for p in self.population:
            if self.best is None or self.best.fit < p.fit:
                self.best = p

            if self.worst is None or self.worst.fit > p.fit:
                self.worst = p

        self.average = self.populationFitness/self.populationSize


    def popFitness(self):
        """ Updates GA's total fitness """

        self.populationFitness = 0

        for x in self.population:
            self.populationFitness += x.fit


    def mating(self):
        """ perform crossover to make the kids """
        self.selection()
        children = []

        # randomly select parents for mating
        for i in range(self.populationSize//4):
            parent1 = np.random.choice(self.population)
            parent2 = np.random.choice(self.population)

            while parent1 is parent2:
                parent2 = np.random.choice(self.population)

            child1, child2 = self.crossover(parent1, parent2)

            children.append(child1)
            children.append(child2)

        # odd population size, get an extra child #PromNightDumpsterBaby
        if self.populationSize % 2 == 1:
            parent1 = np.random.choice(self.population)
            parent2 = np.random.choice(self.population)

            while parent1 is parent2:
                parent2 = np.random.choice(self.population)

            child1, child2 = self.crossover(parent1, parent2)

            children.append(child1)

        for child in children:
            self.population.append(child)


    def selection(self):
        """ Set population to the top 50% of current population """
        sortedList = sorted(self.population, key=lambda Point: Point.fit, reverse=True)
        self.population = []

        for i in range(self.populationSize//2):
            self.population.append(sortedList[i])


    def crossover(self, parent1, parent2):
        """ children are the point in between parent 1 and 2 with a perturbation
        :param parent1:
        :param parent2:
        :return: (child1, child2)
        """

        x = (parent1.x + parent2.x)/2
        y = (parent1.y + parent2.y)/2

        stdDevX = (abs(parent1.x - parent2.x))
        stdDevY = (abs(parent1.y - parent2.y))

        #stdDevX = (self.domain[1] - self.domain[0])/12
        #stdDevY = (self.range[1] - self.range[0])/12

        child1 = None
        child2 = None

        while True:

            x1 = x + np.random.normal(0, stdDevX)
            y1 = y + np.random.normal(0, stdDevY)

            if self.domain[0] <= x1 <= self.domain[1] and self.range[0] <= y1 <= self.range[1]:
                fit = self.func([x1, y1])
                child1 = Point(x1, y1, fit)
                break

        while True:

            x1 = x + np.random.normal(0, stdDevX)
            y1 = y + np.random.normal(0, stdDevY)

            if self.domain[0] <= x1 <= self.domain[1] and self.range[0] <= y1 <= self.range[1]:
                fit = self.func([x1, y1])
                child2 = Point(x1, y1, fit)
                break

        return child1, child2



    def mutate(self):
        """
        Goes through each individual of the population, and with a certain probability will mutate them
        Mutate will move current point to a random location within the valid domain and range
        """

        for mutant in self.population:

            roll = np.random.random_sample()

            if roll <= self.mutationRate:

                x = np.random.uniform(self.domain[0], self.domain[1])
                y = np.random.uniform(self.range[0], self.range[1])
                fit = self.func([x, y])

                mutant.x = x
                mutant.y = y
                mutant.fit = fit




    def run(self):

        self.initializePopulation()

        count = 0

        while count < self.iterations:

            self.evaluate()

            self.selection()

            self.mating()

            self.mutate()

            self.popHistory.append(self.population)

            count += 1

        self.evaluate()

        print()
        print('FINISHED!')
        print('Best fitness value:\t\t',self.best.fit,'\t@',[self.best.x, self.best.y])
        print('Worst fitness value:\t',self.worst.fit,'\t@',[self.worst.x, self.worst.y])
        print('Average fitness value:\t', self.average)

    def getData(self, gen):
        """ returns x and y data for a given generation """
        x = []
        y = []

        for a in self.popHistory[gen]:
            x.append(a.x)
            y.append(a.y)

        return x, y


    def graph(self):
        fig = plt.figure()
        plt.xlim(self.domain[0], self.domain[1])
        plt.ylim(self.range[0], self.range[1])

        graph, = plt.plot([], [], 'o')

        def animate(i):
            x, y = self.getData(i)
            graph.set_data(x, y)
            return graph

        t1 = np.linspace(self.domain[0], self.domain[1], 100)
        t2 = np.linspace(self.range[0], self.range[1], 100)

        X, Y = np.meshgrid(t1, t2)

        Z = self.func([X, Y])

        plt.contour(X, Y, Z, 100, cmap='gist_ncar', alpha=0.3)

        ani = FuncAnimation(fig, animate, frames=self.iterations, interval=300)
        # ani.save('genetic-algorithm.gif', dpi=80, writer='imagemagick')

        plt.show()


class Point:

    def __init__(self, x, y, fit):

        self.x = x
        self.y = y
        self.fit = fit

########################################################################################################################





