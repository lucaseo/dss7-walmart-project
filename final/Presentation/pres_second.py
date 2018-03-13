from jw_package import *
from functools import *

each_station = []

for station_nbr in range(1,21):
    file_path = 'data/weather_best_close_final/{:02}.csv'.format(station_nbr)
    station = pd.read_csv(file_path,index_col=0)
    station.dropna(axis=0,how='any',inplace=True)
    each_station.append(station)

weather = reduce((lambda x,y : x.append(y)),each_station)

etc = ['day','month','year','station_nbr']
category = ['sunrise','sunset','codesum']

sales = pd.read_csv('data/train.csv')
keys = pd.read_csv('data/key.csv')

data = sales.merge(keys).merge(weather)

for each in etc+category:
    data.drop(each,axis=1,inplace=True)
    
data = data[data['units']!=0]  
data.reset_index(drop = True, inplace = True)

codesum = pd.read_csv('data/weather_v1.csv', index_col = 0)
key = pd.read_csv('data/key.csv', index_col = 0)
key.reset_index(inplace = True)
codesum = codesum.merge(key, on = 'station_nbr' )
codesum = codesum.loc[:,['date','store_nbr','is_holiday','codesum']]

def cate_holiday(holiday):
    if holiday == 'holiday' or holiday == 'holiday_work':
            return 1
    else:
        return 0
def cate_codesum(codesum):
    if codesum == 'MO':
        return 0 
    else:
        return 1