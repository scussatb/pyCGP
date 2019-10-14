from collections import Iterable

def change_interval (x, inmin, inmax, outmin, outmax):
    # making sure x is in the interval
    x = max(inmin, min(inmax, x))
    # normalizing x between 0 and 1
    x = (x - inmin) / (inmax - inmin)
    # denormalizing between outmin and outmax
    return x * (outmax - outmin) + outmin

def change_float_to_int_interval (x, inmin, inmax, outdiscmin, outdiscmax):
    x = change_interval(x, inmin, inmax, 0, 1)
    return round(x * (outdiscmax - outdiscmin) + outdiscmin)

