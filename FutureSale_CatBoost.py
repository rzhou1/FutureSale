'''Future retailer sale prediction with time-series dataset.
Here we demonstrate a catboost solution.'''

import pandas as pd
import numpy as np
import sklearn.metrics
import seaborn as sns
from datetime import datetime
from itertools import product
from catboost import Pool, CatBoostRegressor, cv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

#Functions definition for data loading and preprocessing

def salesTrainLoadandPreprocess():
    '''Load sales train file, remove outliers, and imputer/replace invalid data.'''
    data = pd.read_csv('sales_train.csv')
    data = data[data.item_cnt_day<=1000] #remove outliers
    data = data[data.item_price<50000] #remove outliers
    data.loc[(data.item_price<0), 'item_price'] = data.loc[(data.date_block_num==4)&(data.item_id==2973), 'item_price']\
        .mode() #replace negative item_price (see EDA for more details)
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], format="%d.%m.%Y")
    data['date'] = pd.to_datetime(data['date'])

    return data


def shopsLoad():
    shops = pd.read_feather('sh_en.feather')
    shops = shops.rename(columns={'sname': 'shop_name_ru', 'sid': 'shop_id', 'en_sname': 'shop_name'})
    shops['city'] = shops['shop_name'].str.split(' ').apply(lambda x: x[0])
    shops.loc[(shops.city == 'Outbound') | (shops.city == 'Digital'), 'city'] = 'Online'
    shops.loc[(shops.shop_id == 34) | (shops.shop_id == 35), 'city'] = 'Nizhny Novgorod'
    shops.loc[shops.city == 'SPb', 'city'] = 'St. Petersburg'

    return shops


def itemsLoad():
    items = pd.read_feather('it_en.feather')
    items = items.rename(columns={'it_name': 'item_name_ru', 'it_id': 'item_id', 'itc_id': 'item_category_id',
                                  'en_it_name': 'item_name'})

    return items


def categoryLoadAndPreprocess():
    '''Load categories file, extract and preprocess features.'''

    categories = pd.read_feather('itc_en.feather')
    categories = categories.rename(
        columns={'itc_name': 'item_category_name_ru', 'itc_id': 'item_category_id', 'en_itc_name': 'item_category'})

    extractCatFeatures = ['music', 'game', 'book', 'gifts', 'movie', 'payment cards', 'programs', 'ps', 'xbox',
                          'accessories', 'consoles', 'software', 'office', 'tickets', 'delivery']
    categories['item_category'] = categories['item_category'].str.lower()

    for f in extractCatFeatures:
        categories[f] = categories['item_category'].str.contains(f)
        categories[f] = categories[f].astype(int)

    categories['games'] = categories['game'] + categories['ps'] + categories['xbox'] - 2 * categories[
        'accessories'] - 2 * categories['consoles']  # classify 'ps' and 'xbox' as games category
    categories['game accessories'] = categories['accessories'] + categories[
        'consoles']  # classifty 'consoles' as accessories
    categories['programs'] = categories['programs'] + categories['software'] + categories[
        'office']  # classify 'software' and 'office' as program
    categories.loc[categories.games > 1, 'games'] = 1
    categories.loc[categories.games < 0, 'games'] = 0
    categories.loc[categories.accessories > 1, 'game accessories'] = 1
    categories.loc[categories.programs > 1, 'programs'] = 1

    redudantFeatures = ['movie', 'music', 'games']  # category_id 32 classified as payment cards category only

    for feature in redudantFeatures:
        categories.loc[categories.item_category_id == 32, feature] = 0

    categories.loc[categories.item_category_id == 35, 'games'] = 0  # payment cards only.

    categoryIds = [0, 81, 82, 83]  # classified as electronic accessories

    categories['electronic accessories'] = 0

    for id in categoryIds:
        categories.loc[categories.item_category_id == id, 'electronic accessories'] = 1

    categories = categories.drop(['game', 'ps', 'xbox', 'consoles', 'software', 'office', 'accessories'], axis=1)

    newCatFeatures = [col for col in categories.columns.tolist() if 'category' not in col]

    categories['category_nid'] = 0

    for j, item in enumerate(newCatFeatures):
        categories.loc[:, 'category_nid'] += categories.loc[:, item] * (j + 1)

    newCats = pd.Series(newCatFeatures)
    categories['category'] = (categories['category_nid'] - 1).map(newCats)

    dropfeatures = [col for col in categories.columns.tolist() if 'category' not in col]

    categories = categories.drop(dropfeatures, axis=1)
    categories = categories.drop(['item_category_name_ru', 'item_category', 'category_nid'], axis=1)

    return categories

#Merge original datasets of sales, items, shops, and categories into train
def trainDataMerge(sales, items, shops, categories):
    '''Merge train datasets of all categories.'''
    train = pd.merge(sales, items, on='item_id', how='left')
    train = pd.merge(train, shops, on='shop_id', how='left')
    train = pd.merge(train, categories, on='item_category_id', how='left')
    train = train.rename(columns={'item_category_id': 'category_id', 'item_category': 'category_name'})
    train = train.drop(['item_name_ru', 'shop_name_ru'], axis=1)

    item_categories = items.merge(categories, how='left', on=['item_category_id'])
    item_categories = item_categories.drop(['item_name_ru'], axis=1)
    item_categories = item_categories.rename(
        columns={'item_category_id': 'category_id', 'item_category': 'category_name'})

    return train, item_categories


def replaceShopIdDuplicates(data):
    '''Replace duplicated shops.'''
    data.loc[data.shop_id == 57, 'shop_id'] = 0
    data.loc[data.shop_id == 58, 'shop_id'] = 1
    data.loc[data.shop_id == 11, 'shop_id'] = 10
    data.loc[data.shop_id == 40, 'shop_id'] = 39

    return data

#The target is to predict retailers' sales by month
#Functions definitions for aggregating train data by month and adding features
def aggbyMonth(data):
    '''Aggregate train sales by month and clip it [1%, 99%] to minimize outliers' effect.'''

    col_index = ['shop_id', 'item_id', 'date_block_num']
    grid = []
    for date_block in data['date_block_num'].unique():
        shop_month = data[data['date_block_num']==date_block]['shop_id'].unique()
        item_month = data[data['date_block_num']==date_block]['item_id'].unique()
        grid.append(np.array(list(product(*[shop_month, item_month, [date_block]])), dtype='int32'))

    grid = pd.DataFrame(np.vstack(grid), columns = col_index)

    train_m = grid.merge(item_categories, how='left', on=['item_id'])
    train_m = train_m.drop(['item_name'], axis=1)

    agg_items_m = data.groupby(col_index, as_index=False).agg({'item_cnt_day': {'item_cnt_M_sum': 'sum',
                                                                                'item_cnt_M_avg': 'mean'},
                                                                'item_price': {'item_price_M_avg': 'mean'}})
    agg_items_m.columns = [col[0] if col[-1]=='' else col[-1] for col in agg_items_m.columns.values]
    train_m = pd.merge(train_m, agg_items_m, how='left', on=col_index).fillna(0)
    train_m['item_cnt_M_sum'] = train_m['item_cnt_M_sum'].clip(0,40)
    train_m['item_cnt_M_avg'] = train_m['item_cnt_M_avg'].clip(0,20)

    return train_m


def percentile(q):
    def percentile_(a):
        return np.percentile(a, q)

    percentile_.__name__ = 'percentile_%d' % q
    return percentile_


def aggItemsMonthly(data_daily, data_monthly, items):
    aggs = []
    for i in items:
        agg = data_daily.groupby(['date_block_num', '{}'.format(i)], as_index=False).agg(
            {'item_cnt_day': {'{}_item_cnt_M_sum'.format(i): 'sum',
                              '{}_item_cnt_M_avg'.format(i): 'mean',
                              '{}_item_cnt_M_10'.format(i): percentile(10),
                              '{}_item_cnt_M_25'.format(i): percentile(25),
                              '{}_item_cnt_M_75'.format(i): percentile(75),
                              '{}_item_cnt_M_90'.format(i): percentile(90)},
             'item_price': {'{}_item_price_M_avg'.format(i): 'mean', }})
        agg.columns = [col[0] if col[-1] == '' else col[-1] for col in agg.columns.values]
        aggs.append(agg)

    for i, item in enumerate(items):
        data_monthly = pd.merge(data_monthly, aggs[i], how='left', on=['date_block_num', '{}'.format(item)]).fillna(0)

    return data_monthly

#Function to define test data
def testDefine(item_categories, train_m):
    '''Define test data.'''
    test = pd.read_csv('test.csv')  # load test file, contains 'ID', 'shop_id', 'item_id'.
    test['date_block_num'] = 34  # add 'date_block_num'
    test = test.merge(item_categories, how='left', on=['item_id']).drop(['item_name'], axis=1)  # add category_id
    price_cols = [col for col in train_m.columns.tolist() if 'price' in col]
    price_cols.append('shop_id')
    price_cols.append('item_id')
    train_m_price = train_m[price_cols]
    train_m_price = train_m_price.drop_duplicates(subset=['shop_id', 'item_id'], keep='last')
    test = pd.merge(test, train_m_price, how='left', on=['shop_id', 'item_id']).fillna(0)

    cnt_cols = [col for col in train_m.columns.tolist() if 'cnt' in col]

    for col in cnt_cols:
        test[col] = 0
    cols = test.columns.tolist()
    cols = cols[:7] + cols[10:12] + cols[7:8] + cols[12:18] + cols[8:9] + cols[18:24] + cols[9:10] + cols[24:]
    test = test[cols]

    return test

#Add more features to defined [train, test] data
#Feature 'Revenues' by item, shop, and category
class Revenues(object):
    def __init__(self):
        pass

    def _shopItemRevenue(self, data):
        data['shop_item_revenue_M'] = np.multiply(data['item_price_M_avg'], data['item_cnt_M_sum'])
        return data

    def _shopRevenue(self, data):
        data['shop_revenue_M'] = np.multiply(data['shop_id_item_price_M_avg'], data['shop_id_item_cnt_M_sum'])
        return data

    def _itemRevenue(self, data):
        data['item_revenue_M'] = np.multiply(data['item_id_item_price_M_avg'], data['item_id_item_cnt_M_sum'])
        return data

    def _categoryRevenue(self, data):
        data['category_revenue_M'] = np.multiply(data['category_item_price_M_avg'], data['category_item_cnt_M_sum'])
        return data

    def transform(self, data):
        data = self._shopItemRevenue(data)
        data = self._shopRevenue(data)
        data = self._itemRevenue(data)
        data = self._categoryRevenue(data)

        return data


class RevenueScale(object):
    def __init__(self):
        pass

    def revenueMean(self, data):
        agg = data.groupby(['shop_id', 'item_id']).agg({'shop_item_revenue_M': {'shop_item_revenue_M_avg': 'mean'}})
        agg.columns = [col[-1] for col in agg.columns.values]
        data = data.merge(agg, how='left', on=['shop_id', 'item_id']).fillna(0)

        rev_items = [('shop_id', 'shop_revenue_M'), ('item_id', 'item_revenue_M'), ('category', 'category_revenue_M')]
        mean_revs = []

        for (i, r) in rev_items:
            mean_rev = data.groupby('{}'.format(i)).agg({'{}'.format(r): {'{}_avg'.format(r): 'mean'}})
            mean_rev.columns = [col[-1] for col in mean_rev.columns.values]
            mean_revs.append(mean_rev)

        for i, (s, r) in enumerate(rev_items):
            data = data.merge(mean_revs[i], how='left', on='{}'.format(s)).fillna(0)

        return data

    def scale(self, data):
        items = ['shop_item_revenue_M', 'shop_revenue_M', 'item_revenue_M', 'category_revenue_M']

        for i in items:
            data['{}_delta'.format(i)] = (1.0 * (data[i] - data['{}_avg'.format(i)]) / data['{}_avg'.format(i)]).fillna(
                0)

        cols_rev_drop = [col for col in data.columns.tolist() if ('revenue_M' in col) and ('_delta' not in col)]
        data = data.drop(cols_rev_drop, axis=1)

        return data

    def transform(self, data):
        data = self.revenueMean(data)
        data = self.scale(data)

        return data


#Feature: month interval since the first sale for shop-item and item

def firstSaleInterval(data):
    data['shop_item_first_sale'] = data['date_block_num'] - data.groupby(['shop_id', 'item_id'])[
        'date_block_num'].transform('min')

    data['item_first_sale'] = data['date_block_num'] - data.groupby(['item_id'])['date_block_num'].transform('min')

    return data

#Feature: month interal since the last sale for shop-item and item
def lastSaleInterval(data, features):

    cache = {}

    for f in features:
        data[f] = 0
        if f == 'item_last_sale':
            for i, row in data.iterrows():
                key = row.item_id

                if key not in cache:
                    if row.item_cnt_M_sum != 0:
                        cache[key] = row.date_block_num
                else:
                    last_sale_block = cache[key]
                    if row.date_block_num > last_sale_block:
                        data.at[i, f] = row.date_block_num - last_sale_block
                        if row.item_cnt_M_sum != 0:
                            cache[key] = row.date_block_num

        elif f == 'shop_item_last_sale':
            for i, row in data.iterrows():
                key = str(row.shop_id) + '_' + str(row.item_id)

                if key not in cache:
                    if row.item_cnt_M_sum !=0:
                        cache[key] = row.date_block_num
                else:
                    last_sale_block = cache[key]
                    if row.date_block_num > last_sale_block:
                        data.at[i, f] = row.date_block_num - last_sale_block
                        if row.item_cnt_M_sum != 0:
                            cache[key] = row.date_block_num

    return data

#Function for lag features creation

def lagFeatures(data):
    months_lag=[1,2,3,4,5,6,12]
    cols_lag=data.columns.tolist()[9:]
    for col in cols_lag:
        for lag in months_lag:
            data[col+'_lag{}'.format(lag)] = data.groupby(['shop_id', 'item_id'])[col].shift(lag).fillna(0)
    return data

#Add more time-related features
class DateTime(object):

    def __init__(self):
        pass

    def _year(self, data):
        data.loc[data.date_block_num < 12, 'year'] = 2013
        data.loc[(data.date_block_num > 11) & (data.date_block_num < 24), 'year'] = 2014
        data.loc[data.date_block_num > 23, 'year'] = 2015
        data['year'] = data['year'].astype(int)

        return data

    def _month(self, data):
        data['month'] = data['date_block_num'] % 12 + 1

        return data

    def _day(self, data):
        data['day'] = 1

        return data

    def _date(self, data):
        data['date'] = pd.to_datetime(data[['year', 'month', 'day']])

        return data

    def _daysPerMonth(self, data):
        days = pd.Series([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        data['days_per_month'] = data['month'].map(days) / 30

        return data

    def transform(self, data):
        #         data = self._year(data)
        data = self._month(data)
        #         data = self._day(data)
        #         data = self._date(data)
        data = self._daysPerMonth(data)

        return data

#Function for train, val and test data splits

def trainTestSplit(data):
    '''Train and test data can be split by date_block_num.
    Tweleve training datasets stored in X_train and Y_train.'''

    X_train = []
    y_train = []
    for block_num in np.arange(22, 34, 1):
        X_train_block = data[data['date_block_num'] < block_num].iloc[:, 3:-1]
        y_train_block = data[data['date_block_num'] < block_num].iloc[:, -1]

        X_train.append(X_train_block)
        y_train.append(y_train_block)

    X_val = data[data['date_block_num'] == 33].iloc[:, 3:-1]
    y_val = data[data['date_block_num'] == 33].iloc[:, -1]

    test = data[data['date_block_num'] == 34].iloc[:, 3:-1]

    return X_train, y_train, X_val, y_val, test

#catboost modeling
#define eval metrics
def rmse(y_actual, y_pred):
    '''Metrics - Root Mean Squared Error.
    y_actual: vector of actual values;
    y_pred: vector of test values.'''

    return np.sqrt(sklearn.metrics.mean_squared_error(y_actual, y_pred))

def catboostModel(X_train, Y_train, X_val, y_val, test, cat_features):
    '''catboost modeling.'''
    scores = []
    pred = []
    for i in range(len(Y_train)):
        train_pool = Pool(X_train[i], Y_train[i], cat_features=cat_features)
        cbr = CatBoostRegressor(iterations=500, learning_rate=0.3, depth=5, l2_leaf_reg=10, loss_function='RMSE',
                                random_seed=0)
        cbr.fit(train_pool)
        y_hat = cbr.predict(X_val)
        y_pred = cbr.predict(test)
        scores.append(rmse(y_val, y_pred))
        pred.append(y_pred)

    return scores, pred

def submission(data):
    '''predicted mean values for submission.'''
    sub = pd.DataFrame(data={'ID': test.ID, 'item_cnt_month': data.mean()})
    sub.to_csv('sub_mean.csv', index=False)

    return sub

if __name__ == '__main__':

    #Part 1: Daily train data loading and preprocessing
    sales_train = salesTrainLoadAndPreprocess()
    shops = shopsLoad()
    items = itemsLoad()
    categories = categoryLoadAndPreprocess()
    train, item_categories = trainDataMerge(sales_train, items, shops, categories)
    train = replaceShopIdDuplicates(train)

    #Part 2: Monthly train and test data creation and preprocessing
    train_m = aggbyMonth(train)
    train_m = aggItemsMonthly(train, train_m, ['shop_id', 'item_id', 'category'])

    test = testDefine(item_categories, train_m)
    test = replaceShopIdDuplicates(test)
    data_all_m = pd.concat([train_m, test.drop('ID', axis=1)]).drop('category_id', axis=1)
    data_all_m = data_all_m.merge(shops.drop(['shop_name_ru', 'shop_name'], axis=1))

    #Part 3: Feature engineering for train and test data
    data_all_m = Revenues().transform(data_all_m)
    data_all_m = RevenueScale().transform(data_all_m)
    data_all_m = FeaturesTransform().transform(data_all_m)
    data_all_m = firstSaleInterval(data_all_m)
    data_all_m = lastSaleInterval(data_all_m, ['item_last_sale', 'shop_item_last_sale'])

    cols = data_all_m.columns.tolist()
    cols = cols[0:4] + cols[28:29] + cols[33:] + cols[29:33] + cols[4:28]
    data_all_m = data_all_m[cols]# re-organize columns for the coming creation of lag features.
    data_all_m = lagFeatures(data_all_m)

    data_all_m = DateTime().transform(data_all_m)

    #Part 4: train, val, test data split
    cols = data_all_m.columns.tolist()
    cols = cols[0:9] + cols[37:] + cols[15:16]
    data_all_m = data_all_m[cols]#re-organize columns for defining model-learnable train and test.


    X_train, y_train, X_val, y_val, test = trainTestSplit(data_all_m) #train and test data split and define for modeling;
    # 12 training datasets created for generating 12 predictions (and models). The final prediciton is the average of the 12 predictions.

    #Part 5: model and submission
    cat_features = [0,1]
    scores, pred = catboostModel(X_train, y_train, X_val, y_val, test, cat_features)

    sub = submission(pred)

    print scores

