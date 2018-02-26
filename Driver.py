import GA
import functions as fun


if __name__ == '__main__':
    gen = GA.GA(fun.peaks, [-5, 5], [-5, 5], 30, 60, .05)
    gen.run()
    gen.graph()


