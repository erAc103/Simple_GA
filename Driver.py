import GA
import functions as fun
import csv


if __name__ == '__main__':
    gen = GA.GA(fun.sinc3D, [-5, 5], [-5, 5], 15, 20, .01)
    gen.run()
    gen.graph()


