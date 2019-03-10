from math import sqrt
from random import randint


def evaluate_and(individual, nbr=100):
    total = 0
    for _ in range(nbr):
        a = randint(0, 1)
        b = randint(0, 1)
        c = bool(a) and bool(b)
        if (round(individual(a, b)[0]) % 2 == 0) is c:
            pass
            # print(individual)
            # print(a, b, c)
        else:
            total += 1
    return total / nbr


def evaluate_sum(individual, nbr=10):
    total = 0
    for _ in range(nbr):
        a = randint(-100, 100)
        b = randint(-100, 100)
        c = a + b
        if not c == round(individual(a, b)[0]):
            total += 1
    return total / nbr


def evaluate_sqrt(individual, nbr=300):
    total = 0
    for _ in range(nbr):
        a = randint(0, 1000)
        c = round(sqrt(a))
        total += abs(c - individual(a)[0])
    return total / nbr
