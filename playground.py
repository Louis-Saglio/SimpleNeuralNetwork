from random import randint
from statistics import mean

from evolution import get_random_nt

gen_nbr = 2000

# generate
pop_size = 200
nts = get_random_nt(pop_size)


def evaluate(nt, nbr=10):
    total = 0
    for _ in range(nbr):
        a = randint(0, 100)
        b = randint(0, 100)
        c = a + b
        total += abs(nt.run((a, b))[0] - c)
    return total / nbr


for _ in range(gen_nbr):
    try:
        # select
        nts = nts[: pop_size // 2]
        nts += [nt.copy() for nt in nts]
        # mutate
        [nt.mutate() for nt in nts if randint(0, 100) == 0]
        # sort
        nts = sorted(nts, key=lambda nt: evaluate(nt))

        scores = [evaluate(nt) for nt in nts]
        print(round(mean(scores), 2), round(min(scores), 2))
    except:
        if input("Continue ? (y/n)") == 'n':
            break
