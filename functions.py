import numpy as np

""" Test Functions """


def sinc3D(x):
    return np.sinc(x[0])*np.sinc(x[1])


def peaks(x):
    return 3*((1-x[0])**2)*np.exp(-1*(x[0]**2) - (x[1]+1)**2) - 10*((x[0]/5) - x[0]**3 - x[1]**5)\
            * np.exp(-1*x[0]**2-x[1]**2) - (np.exp(-1*(x[0]+1)**2 - x[1]**2)/3)


def eosom(x): # inverted easom function
    return -1 * (-1 * np.cos(x[0]) * np.cos(x[1]) * np.exp(-1 * (x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2))


def rastrigin(x):
    return -1*(20 + x[0]**2 + x[1]**2 - 10*(np.cos(np.pi * 2 * x[0]) + np.cos(np.pi * 2 * x[1])))


def beale(x):
    return -1*((1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * (x[1] ** 2)) ** 2 + (
                2.625 - x[0] + x[0] * (x[1] ** 3)) ** 2)