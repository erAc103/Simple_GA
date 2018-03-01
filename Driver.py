import GA
import functions as fun
import csv


if __name__ == '__main__':
    gen = GA.GA(fun.sinc3D, [-5, 5], [-5, 5], 15, 40, .05, prematureStop=True)
    gen.run()
    gen.graph()


