import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import GA
import functions as fun
import numpy as np
import csv


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
    gen = GA.GA(fun.sinc3D, [-15, 15], [-15, 15], 20, 100, 0.05, prematureStop=True)
    gen.run()
    gen.graph()

    ''' stuff for writing and reading data
    for i in range(1,7):
        gen1 = GA.GA(fun.sinc3D, [-8, 8], [-8, 8], i*5, 100, 0.01, prematureStop=True)
        gen2 = GA.GA(fun.sinc3D, [-8, 8], [-8, 8], i*5, 100, 0.005, prematureStop=True)
        saveToCSV('sinc3D_16x16_pop'+str(i*5)+'_mut-01.csv', gen1, 1000)
        saveToCSV('sinc3D_16x16_pop'+str(i*5)+'_mut-005.csv', gen2, 1000)

    popConv1 = []
    popConv2 = []
    for i in range(1,7):
        popConv1.append(loadColumnFromCSV('sinc3D_16x16_pop'+str(i*5)+'_mut-005.csv', 4))
        popConv2.append(loadColumnFromCSV('sinc3D_16x16_pop'+str(i*5)+'_mut-05.csv', 4))

    popConv = popConv1 + popConv2


    for i in range(1,13):
        plt.subplot(2, 6, i)
        plt.ylim(0,280)

        if i < 7:
            plt.title('pop'+str(i*5)+'mut 0.005')
        else:
            plt.title('pop'+str((i-6)*5)+'mut 0.05')

        sns.distplot(popConv[i-1], kde=False, bins=30)

    plt.show()
    

    x = loadColumnFromCSV('sinc3D_16x16_pop15_mut-05.csv', 4)
    y = loadColumnFromCSV('sinc3D_16x16_pop15_mut-005.csv', 4)

    print('0.005 mean =', np.mean(y))
    print('0.05 mean =', np.mean(x))

    '''







