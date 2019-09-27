from collections import Iterable

def change_interval (x, inmin, inmax, outmin, outmax):
    # normalizing x between 0 and 1
    x = (x - inmin) / (inmax - inmin)
    # denormalizing between outmin and outmax
    return x * (outmax - outmin) + outmin