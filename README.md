# [Philippa_&_Ronya]

**Group members:**
- Philippa
- Ronya
- Eduardo

**This repository contains:***

1. Data project
1. Model project
1. Exam project

*All code can be run with a standard Anaconda Distribution for Python 3.13.*

Data Project 

Inequality in Denmark 

import and set magics: 

import numpy as np
import pandas as pd

# APIs
from fredapi import Fred
from dstapi import DstApi

# plotting
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


Question 1.1 

We load the data from Statistikbanken.dk 
IFOR41 = DstApi('IFOR41')

IFOR41.tablesummary(language='en')


