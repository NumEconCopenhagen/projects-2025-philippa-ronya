import pandas as pd
from dstapi import DstApi

# Importing table IFOR41 from Danmarks Statistik
IFOR41 = DstApi('IFOR41')
IFOR41.tablesummary(language='en')

# Retrive Gini for all municipilites and years of table IFOR41
params1 = {
    'table': 'IFOR41',
    'format': 'BULK', 
    'lang': 'en',
    'variables': [
        {'code': 'ULLIG', 'values': ['70']}, 
        {'code': 'KOMMUNEDK', 'values': ['*']},
        {'code': 'Tid', 'values': ['*']}, 
        ]
    }

gini_municipalities = IFOR41.get_data(params=params1)


# Importing table IFOR32 from Danmarks Statistik
IFOR32 = DstApi('IFOR32')
IFOR32.tablesummary(language='en')

# Retrive Gini for all municipilites and years of table IFOR32
params1 = {
    'table': 'IFOR32',
    'format': 'BULK', 
    'lang': 'en',
    'variables': [
        {'code': 'DECILGEN', 'values': ['*']}, 
        {'code': 'KOMMUNEDK', 'values': ['*']},
        {'code': 'Tid', 'values': ['*']}, 
        ]
    }

income_deciles= IFOR32.get_data(params=params1)

# Summing all deciler for every municipalities and years 
total_income = income_deciles.groupby(['KOMMUNEDK', 'TID'])['INDHOLD'].sum()

# Extract the income of the richest 10 percent
richest = (
    income_deciles[income_deciles['DECILGEN'] == 'Tenth decil']
    .set_index(['KOMMUNEDK', 'TID'])['INDHOLD']
)
# Calculate the top 10 percent share

TOP10_municipalities = richest/total_income * 100
TOP10_municipalities = TOP10_municipalities.reset_index()

#Rename INDHOLD to TOP10
TOP10_municipalities = TOP10_municipalities.rename(
    columns={'INDHOLD': 'TOP10'}
)

