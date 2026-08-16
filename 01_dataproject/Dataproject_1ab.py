# importing necessary packages 
import pandas as pd
from dstapi import DstApi

# Importing table IFOR41 from Danmarks Statistik
IFOR41 = DstApi('IFOR41')

# Getting variable levels for 'ULLIG'
IFOR41.variable_levels('ULLIG',language='en') 

# Creating a dictionary with parameters for the API request
params1 = {
    'table': 'IFOR41',
    'format': 'BULK', 
    'lang': 'en',
    'variables': [
        {'code': 'ULLIG', 'values': ['70']}, 
        {'code': 'KOMMUNEDK', 'values': ['000']},
        {'code': 'Tid', 'values': ['*']}, 
        ]
    }

gini = IFOR41.get_data(params=params1)

# Converting the 'INDHOLD' column to numeric type
gini['INDHOLD'] = pd.to_numeric(gini['INDHOLD'])

# Clean the data and rename columns for clarity (cleaning of ULLIG and INDEHOLD , so we only have TID and GINI
gini_clean = gini[['TID', 'INDHOLD']].copy()
gini_clean = gini_clean.rename(columns={'INDHOLD': 'GINI'})

gini_clean = gini_clean.sort_values('TID')

# Importing table IFOR32 from Danmarks Statistik
IFOR32 = DstApi('IFOR32')
IFOR32.tablesummary(language='en')

# Getting variable levels for 'DECILGEN'
IFOR32.variable_levels('DECILGEN',language='en') 

# Creating a dictionary with parameters for the API request
params2 = {
    'table': 'IFOR32',
    'format': 'BULK', 
    'lang': 'en',
    'variables': [
        {'code': 'DECILGEN', 'values': ['*']}, 
        {'code': 'KOMMUNEDK', 'values': ['000']},
        {'code': 'Tid', 'values': ['*']}, 
        ]
    }

deciles = IFOR32.get_data(params=params2)

# Total income for each year
total_income = deciles.groupby('TID')['INDHOLD'].sum()

# Income of the richest 10 percent
richest = deciles[deciles['DECILGEN'] == 'Tenth decil'].set_index('TID')['INDHOLD']

# Top 10 percent share
TOP10 = richest / total_income * 100

#Using inner join to combine gini_clean and TOP10 into a new DataFrame called merged.
merged = pd.merge(
    gini_clean,
    TOP10,
    on='TID',
    how='inner',
    validate='1:1') 

merged = merged.rename(columns={'INDHOLD': 'TOP10'})

