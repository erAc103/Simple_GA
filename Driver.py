import GA
import functions as fun


if __name__ == '__main__':
    gen = GA.GA(fun.sinc3D, [-5, 5], [-5, 5], 20, 30, .01)
    gen.run()
    gen.graph()

