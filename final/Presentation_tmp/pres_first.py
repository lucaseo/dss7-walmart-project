from jw_package import *

file_path = 'data/weather_v4_delete_nan.csv'

sales = pd.read_csv('data/train.csv')
keys = pd.read_csv('data/key.csv')
raw = pd.read_csv(file_path,index_col=0)

category = ['station_nbr','date','sunset','sunrise','codesum']
for each in category:
    raw.drop(each, axis=1, inplace=True)
    
raw.drop('tavg',axis=1,inplace=True)
raw.drop('sealevel',axis=1,inplace=True)
raw.drop('stnpressure',axis=1,inplace=True)
raw.drop('wetbulb',axis=1,inplace=True)
raw.drop('tmin',axis=1,inplace=True)
raw.drop('avgspeed',axis=1,inplace=True)
raw.drop('tmax',axis=1,inplace=True)

trimmed = raw
raw = pd.read_csv(file_path,index_col=0)
trimmed['station_nbr'] = raw['station_nbr']
trimmed['date'] = raw['date']

data = sales.merge(keys).merge(trimmed)
cols = list(data.columns[3:])
cols.remove('station_nbr')

formula = 'units ~ depart + dewpoint + heat + cool + snowfall + preciptotal + resultspeed + resultdir'