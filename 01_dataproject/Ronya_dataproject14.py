import pandas as pd
from dstapi import DstApi

# Importing relevant information about gini from table IFOR41 from Danmarks Statistik
IFOR41 = DstApi('IFOR41')
IFOR41.tablesummary(language='en')

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

gini_municipalities = gini_municipalities[
    ['KOMMUNEDK', 'TID', 'INDHOLD']
]

gini_municipalities = gini_municipalities.rename(
    columns={'INDHOLD': 'GINI'}
)


# Import average disposable income for municipalities from table INDK101 from Danmarks Statistik
INDKF101 = DstApi('INDKF101')

params2 = {
    'table': 'INDKF101',
    'format': 'BULK',
    'lang': 'en',
    'variables': [
        {'code': 'OMRÅDE', 'values': ['*']},
        {'code': 'ENHED', 'values': ['115']},
        {'code': 'BOLIGARTUD', 'values': ['TOT']},
        {'code': 'INDKOMSTTYPE', 'values': ['100']},
        {'code': 'Tid', 'values': ['*']},
    ]
}

average_income = INDKF101.get_data(params=params2)

# Mearging gini and average disponible income for municipalities 
merged_municipalities_part2 = pd.merge(
    gini_municipalities,
    average_income,
    left_on =['KOMMUNEDK','TID'],
    right_on =['OMRÅDE','TID'],
    how='inner',
    validate='1:1') 

merged_municipalities_part2 = merged_municipalities_part2[
    ['KOMMUNEDK', 'TID', 'GINI', 'INDHOLD']
]

merged_municipalities_part2 = merged_municipalities_part2.rename(
    columns={'INDHOLD': 'AVERAGE_DISPOSABLE_INCOME'}) 