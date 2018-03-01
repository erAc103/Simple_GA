import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

""" Simple genetic algorithm to maximize functions """


class GA:

    def __init__(self, func, domain, range, populationSize, iterations, mutationRate, prematureStop = False):
        """
        :param func: function to maximize
        :param domain: [min, max] x values
        :param range:  [min, max] y values
        :param populationSize: # of points searching
        :param iterations: # of generations
        :param mutationRate: # 0 <= mutation rate <= 1
        :param prematureStop: stops when points converge
        """

        # setting parameters
        self.func = func
        self.domain = domain
        self.range = range
        self.populationSize = populationSize
        self.iterations = iterations
        self.mutationRate = mutationRate
        self.prematureStop = prematureStop

        # holds population of points
        self.population = []

        # holds copies of best and worst points
        self.best = None
        self.worst = None

        # holds average fitness of population
        self.average = None

        # misc. data for plotting and what not
        self.popHistory = []
        self.bestHistory = []
        self.worstHistory = []
        self.averageHistory = []
        self.convergenceData = []

        # number of generations completed
        self.totalGenerations = 0


    def initializePopulation(self):
        """ Create an initial population of points """

        for i in range(self.populationSize):
            x = np.random.uniform(self.domain[0], self.domain[1])
            y = np.random.uniform(self.range[0], self.range[1])
            fit = self.func([x, y])

            self.population.append(Point(x, y, fit))

        self.popHistory.append(self.population)


    def debug(self):
        """ Used for debugging - prints x and y values for each member of the population """
        count = 1
        for a in self.population:
            print(count,'#  ', a.x, a.y)
            count += 1


    def evaluate(self):
        """ Update populations best, worst, and average """
        clonePop = self.copyPop()

        self.best = clonePop[0]
        self.worst = clonePop[0]

        for point in clonePop:
            if point.fit > self.best.fit:
                self.best = point
            if point.fit < self.worst.fit:
                self.worst = point

        self.average = np.mean(self.getFit(self.population))

        self.saveData(clonePop)


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

        stdDevX = (abs(parent1.x - parent2.x))/2
        stdDevY = (abs(parent1.y - parent2.y))/2

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
        """

        xStdDev = (self.domain[1]-self.domain[0])/15
        yStdDev = (self.range[1]-self.range[0])/15

        for mutant in self.population:

            roll = np.random.random_sample()

            if roll <= self.mutationRate:

                flip = np.random.choice([True, False])

                if flip:

                    mutant.x = np.random.normal(mutant.x, xStdDev)
                    mutant.fit = self.func([mutant.x, mutant.y])

                    # check if new x is valid
                    while self.domain[0] > mutant.x or mutant.x > self.domain[1]:
                        mutant.x = np.random.normal(mutant.x, xStdDev)
                        mutant.fit = self.func([mutant.x, mutant.y])

                else:

                    mutant.y = np.random.normal(mutant.y, yStdDev)
                    mutant.fit = self.func([mutant.x, mutant.y])

                    # check if new y is valid
                    while self.range[0] > mutant.y or mutant.y > self.range[1]:
                        mutant.y = np.random.normal(mutant.y, yStdDev)
                        mutant.fit = self.func([mutant.y, mutant.y])

    def run(self):

        self.initializePopulation()
        self.evaluate()

        count = 0
        while count < self.iterations:

            self.selection()

            self.mating()

            self.mutate()

            self.evaluate()

            count += 1

            # stop search when average and best are close
            if self.prematureStop:
                if abs(self.best.fit - self.average) < 0.0005:
                    self.convergenceData.append(count)
                    self.convergenceData.append([self.best.x, self.best.y])
                    self.convergenceData.append(self.best.fit)
                    break

        self.totalGenerations = count

        print()
        print('FINISHED!')
        print('Best fitness value:\t\t',self.best.fit,'\t@',[self.best.x, self.best.y])
        print('Worst fitness value:\t',self.worst.fit,'\t@',[self.worst.x, self.worst.y])
        print('Average fitness value:\t', self.average)

    # clones population - should fix some weird graphing shit
    def copyPop(self):
        newPop = []
        for p in self.population:
            newP = Point(p.x, p.y, self.func([p.x, p.y]))
            newPop.append(newP)

        return newPop


    # helper function for graph
    def getData(self, gen):
        """ returns x and y data for a given generation """
        x = []
        y = []

        for a in self.popHistory[gen]:
            x.append(a.x)
            y.append(a.y)

        return x, y

    def getFit(self, arr):
        """ Returns fitness values for points in arr """
        ret = []

        for x in arr:
            ret.append(x.fit)

        return ret

    # saves data for graphing
    def saveData(self, pop):
        self.bestHistory.append(self.best)
        self.worstHistory.append(self.worst)
        self.averageHistory.append(self.average)
        self.popHistory.append(pop)


    def graph(self):
        # data for graphing
        best = self.getFit(self.bestHistory)
        worst = self.getFit(self.worstHistory)
        average = self.averageHistory

        fig = plt.figure()

        # Shows points converging
        plt.subplot(1,2,1)
        plt.xlim(self.domain[0], self.domain[1])
        plt.ylim(self.range[0], self.range[1])
        graph1, = plt.plot([], [], 'o')

        # adds contours to first subplot
        t1 = np.linspace(self.domain[0], self.domain[1], 100)
        t2 = np.linspace(self.range[0], self.range[1], 100)

        X, Y = np.meshgrid(t1, t2)

        Z = self.func([X, Y])
        plt.contour(X, Y, Z, 100, cmap='gist_ncar', alpha=0.3)

        # Shows fitness data each iteration
        plt.subplot(1,2,2)
        graph2, = plt.plot([], [], '-o', color='blue', label='Best')  # best
        graph3, = plt.plot([], [], '-x', color='red', label='Worst')  # worst
        graph4, = plt.plot([], [], '-.', color='green', label='Average')  # average
        plt.xlim(0, self.totalGenerations)

        ''' Adjust this to fit max fitness for your function!!! '''
        plt.ylim(-.25, 1.05)


        def animate(i):
            x, y = self.getData(i)
            graph1.set_data(x, y)
            graph2.set_data(range(0,i+1), best[:i+1])
            graph3.set_data(range(0,i+1), worst[:i+1])
            graph4.set_data(range(0,i+1), average[:i+1])

            return graph1, graph2, graph3, graph4

        ani = FuncAnimation(fig, animate, frames=self.totalGenerations+1, interval=300, repeat_delay=4000)
        # ani.save('genetic-algorithm.gif', dpi=80, writer='imagemagick')

        plt.show()


class Point:

    def __init__(self, x, y, fit):

        self.x = x
        self.y = y
        self.fit = fit

########################################################################################################################





