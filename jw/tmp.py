from ml_config import *
from jw_package import *

df = df_1()

store = df[df['store_nbr']==1]
weathers = store.columns[2:20]

weathers = ['depart']
for weather in weathers:
    f1 = lambda x: x.replace(' ','')
    f2 = lambda x: x.replace('-','')
    f3 = lambda x: len(x)
    
    feature = store[weather].apply(f1).apply(f2)
    
    missing_index = (feature.str.contains('M') | feature.str.contains('T'))
    
    nothing_index = feature.apply(f3) == 0
    
    feature[missing_index] = np.nan
    feature[nothing_index] = np.nan
    
#     store[weather] = feature
    store.loc[missing_index,weather] = np.nan