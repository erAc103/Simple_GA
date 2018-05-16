import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import GA
import functions as fun
import numpy as np
import csv

''' Dont worry about this stuff'''
def saveToCSV(name, gen, trials):
    with open(name,'w',newline='') as f:
        writer = csv.writer(f)

        for i in range(1, trials+1):
            gen.run()

            writer.writerow([i, gen.convergenceData[1][0], gen.convergenceData[1][1], gen.convergenceData[2], gen.convergenceData[0]])

            gen.clear()

    print(name, 'is ready!')

def loadColumnFromCSV(name, columnNumber):
    x = []
    with open(name) as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(float(row[columnNumber]))

    return x


if __name__ == '__main__':

    # if you want the fitness values to be graphed properly, line 329 of GA must be modified to fit the fitness
    # of the function
    gen = GA.GA(fun.rastrigin, [-5, 5], [-5, 5], 20, 30, 0.001, prematureStop=False)
    gen.run()
    gen.graph()








