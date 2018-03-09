from jw_package import *
pd.options.mode.chained_assignment = None

def missing_to_nan_weather(station_nbr):
    print('missing_to_nan started !')
    
    full_weather = load_weather()
    
    full_weather = full_weather[full_weather['station_nbr']==station_nbr]
    
    # sunrise, sunset, category는 연속형변수가 아니므로 일단 뺌
    weathers = set(list(full_weather.columns)[2:20])
    category = {'sunset','sunrise','codesum'}
    targets = weathers - category
    
    def trim_str(x):
        import re
                            # 실수만 추출
        result = re.findall('(\d+(?:\.\d+)?)',x)
        if result:
            return float(result[0])
        return np.nan
    
    for weather in targets:
        full_weather.loc[:,weather] = full_weather.loc[:,weather].astype(str).apply(trim_str).astype(float)
    
    print('missing_to_nan completed !')
    return full_weather.reset_index(drop=True)

def how_many_missing(nan_weather):
    
    weathers = set(list(nan_weather.columns)[2:20])
    category = {'sunset','sunrise','codesum'}
    targets = weathers - category
    
    how_many = dict()
    total = len(nan_weather)
    
    for weather in targets:
        not_missing = len(nan_weather[weather].dropna())
    
        msg = 'total = '+ str(total) +' '
        msg += ', missing = '+ str(total - not_missing)
        
        how_many[weather] = msg
    
    return how_many

def all_missing_mean(nan_weather, weather):
    # 1. 날씨가 가장 비슷한 다른 station의 weather를 상속받거나
    # 2. 일단 제외하고 계산하거나
    # 3. 전부 missing인 feature를 빼거나
    
#     this_store_nbr = nan_weather.loc[0,'store_nbr']
    this_station_nbr = nan_weather.loc[0,'station_nbr']
    
#     # 1.
#     if not find_store(this_station_nbr):
#         return nan_weather

#     # 2.
# #     this_station = 
     
    
    print('\t',weather,' EJECTED FROM COLUMN')
    return nan_weather.drop(weather,axis=1)

def partial_missing_mean(nan_weather, weather):
    nan_feature = nan_weather[weather]

    nan_feature[nan_feature.isnull()] = nan_feature.mean()

    nan_weather.loc[:,weather] = nan_feature
    
    print('\t',weather,' : ',nan_feature.mean())
    
    if len(nan_feature[nan_feature.isnull()])==0:
        return nan_weather
    
    print('Error occured when partial_missing() by mean')
    print('station_nbr: ',nan_weather['station_nbr'][0],', feature = ',weather)
    return nan_weather

def replace_by_mean(nan_weather, weather):
    total = len(nan_weather)
    not_missing = len(nan_weather[weather].dropna())
    
    # 1. 아예 없다
    # 같은 station의 store의 평균
    if not_missing == 0:
        return all_missing_mean(nan_weather, weather)
        
    # 2. 있긴 있다
    # 그 store의 평균
    elif 0 < not_missing < total:
        return partial_missing_mean(nan_weather, weather)
    
    # missing이 없는 경우는 아무것도 안함
    return nan_weather

def filling_missing_by_mean(station_nbr):
    nan_weather = missing_to_nan_weather(station_nbr)
    how_many = how_many_missing(nan_weather)
    
    weathers = set(list(nan_weather.columns)[2:20])
    category = {'sunset','sunrise','codesum'}
    targets = weathers - category

    spare_columns = {each : nan_weather[each] for each in list(category)}
    
    # 일단 날리고
    for each in list(category):
        nan_weather.drop(each, axis=1, inplace=True)
    
    # 복원한 다음에
    for weather in targets:
        nan_weather = replace_by_mean(nan_weather,weather)

    # 다시 붙임
    for each in list(category):
        nan_weather[each] = spare_columns[each]
        
    return nan_weather, how_many

def replace_by_closest(nan_weather, weather):
    total = len(nan_weather)
    not_missing = len(nan_weather[weather].dropna())
    
    # 1. 아예 없다
    # 다른 station 중 가장 비슷한 feautre를 가진 애
    if not_missing == 0:
        return all_missing_closest(nan_weather, weather)
        
    # 2. 있긴 있다
    # station내에서 가장 비슷한 feature
#     elif 0 < not_missing < total:
#         return partial_missing(nan_weather, weather)
    
    # missing이 없는 경우는 아무것도 안함
    return nan_weather

def all_missing_closest(nan_weather, weather):
    # 1. 날씨가 가장 비슷한 다른 station의 weather를 상속받거나
    # 2. 일단 제외하고 계산하거나
    # 3. 전부 missing인 feature를 빼거나
    
#     this_store_nbr = nan_weather.loc[0,'store_nbr']
    this_station_nbr = nan_weather.loc[0,'station_nbr']
    
#     # 1.
#     if not find_store(this_station_nbr):
#         return nan_weather

#     # 2.
# #     this_station = 
     
    
    print('\t',weather,' EJECTED FROM COLUMN')
    return nan_weather.drop(weather,axis=1)

def drop_all_missing(station_nbr):
    nan_weather = missing_to_nan_weather(station_nbr)
    how_many = how_many_missing(nan_weather)
    
    weathers = set(list(nan_weather.columns)[2:20])
    category = {'sunset','sunrise','codesum'}
    targets = weathers - category
    
    spare_columns = {each : nan_weather[each] for each in list(category)}
    
    for weather in targets:
        nan_weather = replace_by_closest(nan_weather,weather)
    
    for each in list(category):
        nan_weather.drop(each, axis=1, inplace=True)
        
    return nan_weather, how_many, spare_columns

def filling_missing_by_closest(station_nbr):
    # 빼기 전 all_missing인 feature,
    # sunset,sunrise,codesum과 같은 비연속 데이터는 제외
    
    weather, how, spare_columns = drop_all_missing(station_nbr)
    
    # intact한 row들
    no_nan = weather.dropna(axis=0, how='any')
    nan_index = list(set(weather.index) - set(no_nan.index))
    
    # missing이 하나라도 있는 row들
    nan = weather.loc[nan_index, :]

    feature = list(weather.columns)[2:-3]

    table = weather.describe()

#     print(feature)

    print('\t','filling by best_close processing..')

    for each in nan.index:
        nan_row = nan.loc[each, :]
        nan_z = pd.Series([(nan_row[f] - table[f]['mean']) /
                           table[f]['std'] for f in feature])
        # 얘가 nan이면 그 feature는 missing value임

        distance = dict()

        for idx in no_nan.index:
            intact = no_nan.loc[idx, :]
            intact_z = pd.Series(
                [(intact[f] - table[f]['mean']) / table[f]['std'] for f in feature])

            diff = abs(nan_z - intact_z)
            diff = diff.mean()

            distance[idx] = diff
            
        best_close_idx = min(distance, key=distance.get)
        missing_feature = nan_row.isnull()

        nan_row[missing_feature] = no_nan.loc[best_close_idx, missing_feature]
        weather.loc[each, :] = nan_row

    print('\t','filling by best_close finished !')
        
    weather['codesum'] = spare_columns['codesum']
    weather['sunrise'] = spare_columns['sunrise']
    weather['sunset'] = spare_columns['sunset']
        
    return weather, how

def older_filling_missing_by_closest(station_nbr):
    # 빼기 전 all_missing인 feature,
    # sunset,sunrise,codesum과 같은 비연속 데이터는 제외
    
    weather, how, spare_columns = drop_all_missing(station_nbr)
    
    # intact한 row들
    no_nan = weather.dropna(axis=0, how='any')
    nan_index = list(set(weather.index) - set(no_nan.index))
    
    # missing이 하나라도 있는 row들
    nan = weather.loc[nan_index, :]

    feature = list(weather.columns)[2:-3]

    table = weather.describe()

#     print(feature)

    print('\t','filling by best_close processing..')

    for each in nan.index:
        nan_row = nan.loc[each, :]
        nan_z = pd.Series([(nan_row[f] - table[f]['mean']) /
                           table[f]['std'] for f in feature])
        # 얘가 nan이면 그 feature는 missing value임

        distance = dict()

        for idx in no_nan.index:
            intact = no_nan.loc[idx, :]
            intact_z = pd.Series(
                [(intact[f] - table[f]['mean']) / table[f]['std'] for f in feature])

            diff = abs(nan_z - intact_z)
            diff = diff.mean()

            distance[idx] = diff
            
        best_close_idx = min(distance, key=distance.get)
        missing_feature = nan_row.isnull()

        nan_row[missing_feature] = no_nan.loc[best_close_idx, missing_feature]
        weather.loc[each, :] = nan_row

    print('\t','filling by best_close finished !')
        
    weather['codesum'] = spare_columns['codesum']
    weather['sunrise'] = spare_columns['sunrise']
    weather['sunset'] = spare_columns['sunset']
        
    return weather, how