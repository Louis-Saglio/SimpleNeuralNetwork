from random import randint
from statistics import mean

from evolution import get_random_nt

gen_nbr = 100

# generate
pop_size = 2000
nts = get_random_nt(pop_size)

for _ in range(gen_nbr):
    # select
    nts = nts[: pop_size // 2]
    nts += nts
    # mutate
    [nt.mutate() for nt in nts if randint(0, 100) == 0]
    # sort
    nts = sorted(nts, key=lambda nt: abs(nt.run((42, 13))[0] - 55))

    scores = [abs(nt.run((42, 13))[0] - 55) for nt in nts]
    print(round(mean(scores), 2), round(min(scores), 2))
