import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import sklearn as sk
import matplotlib as mpl
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
sns.set_color_codes()
print('import configuration completed !')
print('Data configuration has been started !')


'''
merge data
'''
sales = pd.read_csv('../data/train.csv')
keys = pd.read_csv('../data/key.csv')
weather = pd.read_csv('../data/weather.csv')
df_1 = pd.merge(weather, keys)
df_1 = pd.merge(df_1, sales)




'''
parse date
'''
def make_year(date):
    return int(date.split('-')[0])

def make_month(date):
    return int(date.split('-')[1])

def make_day(date):
    return int(date.split('-')[2])

df_1['year'] = df_1['date'].apply(make_year)
df_1['month'] = df_1['date'].apply(make_month)
df_1['day'] = df_1['date'].apply(make_day)







print('DataFrame completed !')
