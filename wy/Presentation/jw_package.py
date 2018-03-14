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
from sklearn.datasets import make_regression
sns.set()
sns.set_style("whitegrid")
sns.set_color_codes()

print('import configuration completed !')

def load_weather():
    wther = pd.read_csv('data/weather.csv', parse_dates=['date'])
    dates = wther['date'].dt
    wther['year'] = dates.year
    wther['month'] = dates.month
    wther['day'] = dates.day

    return wther

def load_train_weather(no_zero=True, weather=pd.DataFrame()):
    sales = pd.read_csv('data/train.csv', parse_dates=['date'])
    if no_zero:
        sales = sales.loc[sales['units']!=0,:]
    
    keys = pd.read_csv('data/key.csv')
    
    if len(weather)==0:
        wther = load_weather()
    else:
        wther = weather
    
    
    df_1 = pd.merge(wther, keys)
    
    df_1 = pd.merge(df_1, sales)

    return df_1

def load_final_sample():
    final_sample = pd.read_csv('../data/01. final_sample')
    trimmed = final_sample.iloc[:, 1:]
    
    return (final_sample, trimmed)

print('train + key + weather merging started')
df_1 = load_train_weather(no_zero=False)
print('train + key + weather merging finished')

def find_store(station_nbr):
    '''
    input : station_nbr
    output : store_nbrs dependent to station_nbr
    '''
    
    station = df_1[df_1['station_nbr']==station_nbr]
    return list(station['store_nbr'].unique())

def show_tendency(store_number):
    '''
    input : store_nbr
    output : graph represetning total units for each year, each month
    '''
    
    store_month = df_1.pivot_table(index=['year','month'],columns='store_nbr',values='units',aggfunc=np.sum)
    
    target_store = store_month[store_number]
    that_2012 = target_store.loc[2012]
    that_2013 = target_store.loc[2013]
    that_2014 = target_store.loc[2014]
    
    plt.figure(figsize=(12,8))
    plt.plot(that_2012,label='2012',c='blue',ls='--',lw=4)
    plt.plot(that_2013,label='2013',c='green',ls=':',marker='D',ms=10,lw=4)
    plt.plot(that_2014,label='2014',c='red',lw=4)
    plt.legend(loc='best',prop={'size':20})
    plt.show()

def item_nbr_tendency(store_nbr):
    '''
    input : store_nbr
    output : graph representing units groupped by each year, each month
    '''
    store = df_1[df_1['store_nbr'] == store_nbr]

    pivot = store.pivot_table(index=['year','month'],columns='item_nbr',values='units',aggfunc=np.sum)
    zero_index = pivot==0
    pivot = pivot[pivot!=0].dropna(axis=1,how='all')
    pivot[zero_index]=0
    
    
    pivot_2012 = pivot.loc[2012]
    pivot_2013 = pivot.loc[2013]
    pivot_2014 = pivot.loc[2014]
    
    plt.figure(figsize=(12,8))
    plt.subplot(131)
    sns.heatmap(pivot_2012,cmap="YlGnBu", annot = True, fmt = '.0f')
    plt.subplot(132)
    sns.heatmap(pivot_2013,cmap="YlGnBu", annot = True, fmt = '.0f')
    plt.subplot(133)
    sns.heatmap(pivot_2014,cmap="YlGnBu", annot = True, fmt = '.0f')
    plt.show()

def item_nbr_tendency_finely(store_nbr, year, month_start=-1, month_end=-1, graph=True):
    '''
    input
        1. store_nbr = 스토어 번호
        2. year = 연도
        3. month_start = 시작달
        4. month_start = 끝달
        5. graph = 위의 정보에 대한 item_nbr 그래프 출력여부
    
    output
        1. store_nbr, year, month로 filtering한 item_nbr의 pivot 테이블
    '''
    store = df_1[(df_1['store_nbr'] == store_nbr) &
                 (df_1['year'] == year)]

    if month_start != -1:
        if month_end == -1:
            month_end = month_start + 1
        store = store[(month_start <= store['month']) & (store['month'] < month_end)]

    pivot = store.pivot_table(index='item_nbr',
                              columns='date',
                              values='units',
                              aggfunc=np.sum)

    zero_index = pivot == 0
    pivot = pivot[pivot != 0].dropna(axis=0, how='all')
    pivot[zero_index] = 0

    if graph:
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt='.0f')
        plt.show()

    return pivot

def weather_tendency(store_nbr, year, month_start = -1, month_end = -1):
    '''
    input:
        위와 같음
        
    output:
        위의 정보로 filtering한 train,key,weather DataFrame
    '''
    store = df_1[(df_1['store_nbr'] == store_nbr) &
                 (df_1['year'] == year)]
    
    if month_start!=-1:
        if month_end == -1:
            month_end = month_start + 1
        store = store[(month_start <= store['month']) & (store['month'] < month_end)]
    
    store = store.drop(labels=['item_nbr','units'],axis=1)
#     store = store.iloc[:,:]
    
    store = store.drop_duplicates(keep='first').reset_index(drop=True)
    
    store.index.name='date'
    store.index = store['date']
        
    return store
    
    
def get_correlation(store_nbr, year, month_start=-1, month_end=-1):
    '''
    input:
        위와 같음
    output:
        missing, tracing data를 제외한
        팔린 item_nbr별 각 weather feature에 대한 pearsonr, pvalue를 담은 dictionary
    '''

    '''
    Use-case
        1. 7번 store, 2012년 1월부터 12월까지 item_nbr별 상관관계를 알고 싶다
        get_correlation(7,2012)
        
        2. 7번 store, 2012년 1월부터 3월까지 item_nbr별 상관관계를 알고 싶다
        get_correlation(7,2012,1,3)
        
        3. 7번 store, 2012년 5월만 item_nbr별 상관관계를 알고 싶다
        get_correlation(7,2012,5)
    '''
    correlation = dict()

    units_table = item_nbr_tendency_finely(
        store_nbr, year, month_start, month_end, graph=False)
    weather_table = weather_tendency(store_nbr, year, month_start, month_end)

    weather = list(weather_table.columns[3:20])
    weather.remove('codesum')

    item_nbr = units_table.index

    inner = dict()

    for feature in weather:
        for units in item_nbr:

#             print(feature, units)

            a = weather_table[feature].copy()
            b = units_table.loc[units].copy()

            a = a.apply(lambda x: x.replace(' ', ''))
            a = a.apply(lambda x: x.replace('-', ''))

            missing_index = (a.str.contains('M')) | (a.str.contains('T'))
            nothing_index = a.apply(lambda x: len(x)) == 0

            a[missing_index] = np.nan
            b[missing_index] = np.nan

            a[nothing_index] = np.nan
            b[nothing_index] = np.nan
            
            a.dropna(axis=0, inplace=True)
            b.dropna(axis=0, inplace=True)

            inner[(units, feature, len(a))] = sp.stats.pearsonr(a.astype(float), b)

            correlation[(store_nbr, year, month_start, month_end-1)] = inner

    return correlation

def show_me_pearson(pearson_dict):
    '''
    input:
        get_correlation의 return (상관관계 dictionary)
    output:
        dictionary를 직관적으로 출력함
    '''
    for key,val in pearson_dict.items():
        print(key)
        
        if isinstance(val,dict):
            printee = []
            
            keys = list(val.keys())
        
            keys.sort()
            
            printee += [(k,val[k]) for k in keys]
        
        else:
            printee = val
        
        for each in printee:
            print('\t',each)

def trim_correlation(correlation_dict):
    '''
    input : get_correlation's return
    output : p-value lesser than 0.01, sorted by pearsonr
    '''
    keys = correlation_dict.keys()
    
    result = dict()
    
    for key in keys:
        inner_dict = correlation_dict[key]
        item_nbrs = list(set(each[0] for each in inner_dict.keys()))
        sorted_filtered = filter_sort_inner_dict(inner_dict, item_nbrs)
        
        result[key] = sorted_filtered
        
    return result            

def filter_sort_inner_dict(inner_dict, item_nbrs):
    '''
    input: 
        1. get_correlation[key]
        2. item numbers included in 1
    output: 
        list that filtered and sorted get_correlation[key] 
    '''
    pvalue_filtered = [(key,val) for key,val in inner_dict.items() if val[1] < 0.01]
    
    pearsonr_sorted = []
    
    for item_nbr in item_nbrs:
        pearsonr_sorted += sort_inner_dict(item_nbr,pvalue_filtered)
    
    return pearsonr_sorted

def sort_inner_dict(item_nbr, filtered):
    '''
    input:
        1. item number that gonna sort
        2. get_correlation whose p-value is lesser than 0.01
    output:
        filtered list sorted by pearsonr with item_nbr
    '''
    each_item = [each for each in filtered if each[0][0]==item_nbr]
            
    each_item = sorted(each_item, key = lambda each: abs(each[1][0]), reverse=True)
    
    return each_item

def identify_item_nbr(store_nbr):
    '''
    input: store_nbr
    output: unique item numbers of store_nbr over 2012~2014
    '''
    each_year = [set(item_nbr_tendency_finely(store_nbr,each,graph=False).index) for each in range(2012,2015)]

    for each in each_year:
        unique = set.union(each)
    
    return unique

def columns_to_relation(total_cols, category_cols, y, zero=True):
    '''
        params
            1. total_cols = iteration of independet + dependent variable
            2. category_cols = list of category variable(s)
            3. y = string representing dependent variable
            4. zero = Are you gonna add '+ 0' at the end?
        return
            String representing above relation for sm.OLS.from_formula
    '''
    total_vars = list(total_cols)
    total_vars.remove(y)

    if category_cols:
        category_index = [total_vars.index(each) for each in category_cols]

        for idx in category_index:
            total_vars[idx] = 'C({})'.format(total_vars[idx])

    indep = ' + '.join(total_vars)
    dep = y

    zero = ' + 0' if zero else ''

    return '{} ~ {}{}'.format(dep, indep, zero)

print('function configuration completed ! ')
print('Good to go !')