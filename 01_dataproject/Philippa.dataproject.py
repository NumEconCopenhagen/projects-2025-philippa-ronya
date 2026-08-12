import numpy as np
import pandas as pd


def load_data():
    """
    load data function
    """

    # a. allocate data container
    data = {}

    # b. fill 
    data['GDP'] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    return data

def process_data(data):
    """
    process data function
    """

    # a. verify data
    assert 'GDP' in data, "Data must contain 'GDP' key"

    # b. take log
    for k in ['GDP']:
        
        v = data[k]
        data[f'log_{k}'] = np.log(v)

    return data



# APIs
from fredapi import Fred
from dstapi import DstApi

# plotting
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


IFOR41 = DstApi('IFOR41')

IFOR41.tablesummary(language='en')