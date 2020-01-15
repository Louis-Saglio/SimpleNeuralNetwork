"""
Collection of functions designed to evaluate individuals performance on a specific task
Each function must have one positional argument which is the individual to evaluate
"""


from math import sqrt
from random import randint


def evaluate_and(individual, nbr=100):
    total = 0
    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
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
        total += abs(c - individual(a, b)[0])
    return total / nbr


def evaluate_sqrt(individual, nbr=300):
    total = 0
    for _ in range(nbr):
        a = randint(0, 1000)
        c = round(sqrt(a))
        total += abs(c - individual(a)[0])
    return total / nbr


def evaluate_is_divisible_by(individual, divider=10, nbr=100):
    total = 0
    for _ in range(nbr):
        a = randint(0, 1000)
        c = a % divider == 0
        if (round(individual(a)[0]) % 2 == 0) is c:
            total += 1
    return total / nbr


def evaluate_multiplication(individual, nbr=10):
    total = 0
    for _ in range(nbr):
        a = randint(-100, 100)
        b = randint(-100, 100)
        c = a * b
        total += abs(c - individual(a, b)[0])
    return total / nbr
