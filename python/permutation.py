import numpy as np

def perm(set, n, settilde, ntilde, mode):
    setsettilde = list(zip(list(np.arange(0,n + ntilde)),list(set) + list(settilde)))
    sor = sorted(setsettilde, key=lambda x: x[1])
    swaps, rest = [[i for i, j in sor], [j for i, j in sor]]
    if mode == "col":
        P = np.identity(n + ntilde)[:,swaps]
    elif mode == "row":
        P = np.identity(n + ntilde)[swaps,:]
    return P


