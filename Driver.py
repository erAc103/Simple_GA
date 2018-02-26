import GA
import functions as fun
import numpy as np


if __name__ == '__main__':
    gen = GA.GA(fun.sinc3D, [-5, 5], [-5, 5], 15, 30, .02)
    gen.run()
    gen.graph()


