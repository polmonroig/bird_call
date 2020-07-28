import numpy as np

def count_unique(data):
    """
    Counts the quantity of unique values in a list 
    of values 
    """
    count = {}
    for d in data:
        d = d.split('-')[0]
        if d in count:
            count[d] += 1 
        else:
            count[d] = 1 
    values = [count[k] for k in sorted(count.keys())]
    total = np.arange(len(count))
    return total, values
