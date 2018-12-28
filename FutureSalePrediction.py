'''Future sale prediction with time-series dataset.'''

import pandas as pd
import numpy as np
import sklearn.metrics
import seaborn as sns
from datetime import datetime
from itertools import product
from catboost import Pool, CatBoostRegressor, cv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


def trainDataLoadMerge():
    '''Load train data and merge different data files.
    sales_train: date, date_block_num, shop id, item id, item price, and item_cnt_day (item sale count each day).
    item: item name, item id, item category name and item category id.
    shops: shop name and shop id.
    categories: item category name, item category id.
    train data and item_categories (for test data definition) returned.'''
    sales_train = pd.read_csv('sales_train.csv')
    item_en = pd.read_feather('it_en.feather')
    shops_en = pd.read_feather('sh_en.feather')
    categories_en = pd.read_feather('itc_en.feather')
    item_en = item_en.rename(columns={'it_name': 'item_name_ru', 'it_id': 'item_id', 'itc_id': 'item_category_id',
                                      'en_it_name': 'item_name'})
    shops_en = shops_en.rename(columns={'sname': 'shop_name_ru', 'sid': 'shop_id', 'en_sname': 'shop_name'})
    categories_en = categories_en.rename(columns={'itc_name': 'item_category_name_ru', 'itc_id': 'item_category_id',
                                                  'en_itc_name': 'item_category'})

    train = pd.merge(sales_train, item_en, on='item_id', how='left')
    train = pd.merge(train, shops_en, on='shop_id', how='left')
    train = pd.merge(train, categories_en, on='item_category_id', how='left')
    train = train.rename(columns={'item_category_id': 'category_id', 'item_category': 'category_name'})
    train = train.drop(['item_name_ru', 'shop_name_ru', 'item_category_name_ru'], axis=1)

    item_categories = item_en.merge(categories_en, how='left', on=['item_category_id'])
    item_categories = item_categories.drop(['item_name_ru', 'item_category_name_ru'], axis=1)
    item_categories = item_categories.rename(
        columns={'item_category_id': 'category_id', 'item_category': 'category_name'})
    return train, item_categories

def preprocess(data):
    '''Preprocess daily train data.
    1). remove outliers (too large daily transactions (counts) and too large item price, thresholds are self-defined.
    2). extract date related features.'''

    data = data[data.item_cnt_day<=1000] #remove outliers
    data = data[data.item_price<100000] #remove outliers
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], format="%d.%m.%Y")
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].apply(lambda x: x.month)
    data['year'] = data['date'].apply(lambda x: x.year)
    return data

def aggbyMonth(data):
    '''The final prediction is on aggregating shop and item in monthly basis.
     1). Iterate to generate all shop and item combinations in each month (represented by date_block_num).
     2). Aggregate daily transactions (item_cnt_day) to monthly (item_cnt_M_).'''

    col_index = ['shop_id', 'item_id', 'date_block_num']
    grid = []
    for date_block in data['date_block_num'].unique():
        shop_month = data[data['date_block_num'] == date_block]['shop_id'].unique()
        item_month = data[data['date_block_num'] == date_block]['item_id'].unique()
        grid.append(np.array(list(product(*[shop_month, item_month, [date_block]])), dtype='int32'))

    grid = pd.DataFrame(np.vstack(grid), columns=col_index)

    train_m = grid.merge(item_categories, how='left', on=['item_id'])
    train_m = train_m.drop(['item_name', 'category_name'], axis=1)

    agg_items_m = data.groupby(col_index, as_index=False).agg({'item_cnt_day': {'item_cnt_M_sum': 'sum'},
                                                               'item_price': {'item_price_M_avg': 'mean',
                                                                              'item_price_M_var': 'var'}})
    agg_items_m.columns = [col[0] if col[-1] == '' else col[-1] for col in agg_items_m.columns.values]
    train_m = pd.merge(train_m, agg_items_m, how='left', on=col_index).fillna(0)
    train_m['item_cnt_M_sum'] = train_m['item_cnt_M_sum'].clip(0, 20)

    return train_m

def meanEncode(data, alpha):
    '''Feature extraction: mean encode by smoothing.
    Encode monthly transaction on items by their mean.'''
    global_mean = data['item_cnt_M_sum'].mean()
    item_cnt_mean = data.groupby('item_id')['item_cnt_M_sum'].mean()
    item_cnt_count = data.groupby('item_id')['item_cnt_M_sum'].count()
    smoothing_mean = (np.multiply(item_cnt_mean, item_cnt_count) + alpha * global_mean) / (item_cnt_count + alpha)

    data['item_cnt_M_sm'] = data['item_id'].map(smoothing_mean)
    data['item_cnt_M_sm'].fillna(global_mean, inplace=True)

    return data


def percentile(q):
    '''Define a percentile function to be used together with pandas groupby function.'''
    def percentile_(a):
        return np.percentile(a, q)

    percentile_.__name__ = 'percentile_%d' % q
    return percentile_


def aggItemsMonthly(data_daily, data_monthly, items):
    '''Aggregate category, shop and item by month and calculate their statistics for both counts and prices.'''
    aggs = []
    for i in items:
        agg = data_daily.groupby(['date_block_num', '{}'.format(i)], as_index=False).agg(
            {'item_cnt_day': {'{}_item_cnt_M_sum'.format(i): 'sum',
                              '{}_item_cnt_M_avg'.format(i): 'mean',
                              '{}_item_cnt_M_10'.format(i): percentile(10),
                              '{}_item_cnt_M_25'.format(i): percentile(25),
                              '{}_item_cnt_M_75'.format(i): percentile(75),
                              '{}_item_cnt_M_90'.format(i): percentile(90),
                              '{}_item_cnt_M_var'.format(i): 'var'},
             'item_price': {'{}_item_price_M_avg'.format(i): 'mean',
                            '{}_item_price_M_var'.format(i): 'var'}})
        agg.columns = [col[0] if col[-1] == '' else col[-1] for col in agg.columns.values]
        aggs.append(agg)

    for i, item in enumerate(items):
        data_monthly = pd.merge(data_monthly, aggs[i], how='left', on=['date_block_num', '{}'.format(item)]).fillna(0)

    return data_monthly


def testDefine(item_categories, train_m):
    '''Define test data.'''
    test = pd.read_csv('test.csv')  # load test file, contains 'ID', 'shop_id', 'item_id'.
    test['date_block_num'] = 34  # add 'date_block_num'
    test = test.merge(item_categories, how='left', on=['item_id']).drop(['item_name', 'category_name'], axis=1)  # add category_id
    price_cols = [col for col in train_m.columns.tolist() if 'price' in col]
    price_cols.append('shop_id')
    price_cols.append('item_id')
    train_m_price = train_m[price_cols]
    train_m_price = train_m_price.drop_duplicates(subset=['shop_id', 'item_id'], keep='last')
    test = pd.merge(test, train_m_price, how='left', on=['shop_id', 'item_id']).fillna(0)

    cnt_cols = [col for col in train_m.columns.tolist() if 'cnt' in col]

    for col in cnt_cols:
        test[col] = 0

    cols_test = test.columns.tolist()
    cols_test = cols_test[0:7] + cols_test[13:14] + cols_test[7:9] + cols_test[14:21] + cols_test[9:11] + cols_test[21:28] + cols_test[11:13] + cols_test[28:]

    test = test[cols_test]

    return test


class FeaturesTransform(object):
    '''log transform 'price' features to reduce skewness and clip 'count' features.'''

    def __init__(self):
        pass

    def _featuresClassification(self, data):
        self.price_avg_features = []
        self.price_var_features = []
        self.lower_count_features = []
        self.avg_count_features = []
        self.upper_count_features = []
        self.sum_count_features = []

        cols = data.columns.tolist()
        for col in cols:
            if ('price' in col) & ('avg' in col):
                self.price_avg_features.append(col)
            elif ('price' in col) & ('var' in col):
                self.price_var_features.append(col)
            elif (('cnt' in col) & ('10' in col)) or (('cnt' in col) & ('25' in col)):
                self.lower_count_features.append(col)
            elif ('cnt' in col) & ('avg' in col):
                self.avg_count_features.append(col)
            elif (('cnt' in col) & ('75' in col)) or (('cnt' in col) & ('90' in col)):
                self.upper_count_features.append(col)
            elif (('cnt' in col) & ('sum' in col)) or (('cnt' in col) & ('sm' in col)):
                self.sum_count_features.append(col)
            else:
                pass

        return self.price_avg_features, self.price_var_features, self.lower_count_features, self.avg_count_features, \
               self.upper_count_features, self.sum_count_features

    def _sumFeaturesClassification(self, data):
        self.sum_item_count_features = []
        self.sum_shop_count_features = []
        self.sum_category_count_features = []
        self.sum_item_cnt = []
        for col in self.sum_count_features:
            if 'ItemId' in col:
                self.sum_item_count_features.append(col)
            elif 'ShopId' in col:
                self.sum_shop_count_features.append(col)
            elif 'Category' in col:
                self.sum_category_count_features.append(col)
            else:
                self.sum_item_cnt.append(col)

        return self.sum_item_count_features, self.sum_shop_count_features, self.sum_category_count_features, self.sum_item_cnt

    def _varPriceClip(self, data):
        for col in self.price_var_features:
            if 'ItemId' in col:
                data[col] = data[col].clip(0, 10000)
            elif 'ShopId' in col:
                data[col] = data[col].clip(0, 10000)
            elif 'Category' in col:
                data[col] = data[col].clip(0, 10000)
            else:
                data[col] = data[col].clip(0, 10)

        return data

    def _priceTransform(self, data):
        for col in self.price_avg_features:
            data[col] = np.log1p(data[col])

        for col in self.price_var_features:
            data[col] = np.log1p(data[col])

        return data

    def _lowerCountClip(self, data):
        for col in self.lower_count_features:
            data[col] = data[col].clip(0, 20)

        return data

    def _avgCountClip(self, data):
        for col in self.avg_count_features:
            data[col] = data[col].clip(0, 40)

        return data

    def _upperCountClip(self, data):
        for col in self.upper_count_features:
            data[col] = data[col].clip(0, 60)

        return data

    def _sumItemCountClip(self, data):
        for col in self.sum_item_count_features:
            data[col] = data[col].clip(0, 8000)

        return data

    def _sumShopCountClip(self, data):
        for col in self.sum_shop_count_features:
            data[col] = data[col].clip(0, 10000)

        return data

    def _sumCategoryCountClip(self, data):
        for col in self.sum_category_count_features:
            data[col] = data[col].clip(0, 10000)

        return data

    def _sumCountClip(self, data):
        for col in self.sum_item_cnt:
            data[col] = data[col].clip(0, 40)

        return data

    def _sumCountLogTransform(self, data):
        for col in self.sum_count_features:
            data[col] = np.log1p(data[col])

        return data

    def transform(self, data):
        self.price_avg_features, self.price_var_features, self.lower_count_features, self.avg_count_features, \
        self.upper_count_features, self.sum_count_features = self._featuresClassification(data)
        self.sum_item_count_features, self.sum_shop_count_features, self.sum_category_count_features, \
        self.sum_item_cnt = self._sumFeaturesClassification(data)
        data = self._varPriceClip(data)
        data = self._priceTransform(data)
        data = self._lowerCountClip(data)
        data = self._avgCountClip(data)
        data = self._upperCountClip(data)
        data = self._sumItemCountClip(data)
        data = self._sumShopCountClip(data)
        data = self._sumCategoryCountClip(data)
        data = self._sumCountClip(data)
        #         data = self._sumCountLogTransform(data)

        return data

def lagFeatures(data):
    '''Create lag features for original count features.'''
    months_lag=[1,2,3,4,5,12]
    cols_lag=data.columns.tolist()[12:]
    for col in cols_lag:
        for lag in months_lag:
            data[col+'_lag{}'.format(lag)] = data.groupby(['shop_id', 'item_id'])[col].shift(lag).fillna(0)
    return data


class Revenues(object):
    '''Define revenues based on items, shops, and categories.'''
    def __init__(self):
        pass

    def _shopRevenue(self, data):
        data['Shop_revenue_M'] = np.multiply(data['ShopId_item_price_M_avg'], data['ShopId_item_cnt_M_sum'])
        return data

    def _itemRevenue(self, data):
        data['Item_revenue_M'] = np.multiply(data['ItemId_item_price_M_avg'], data['ItemId_item_cnt_M_sum'])
        return data

    def _categoryRevenue(self, data):
        data['Category_revenue_M'] = np.multiply(data['Category_item_price_M_avg'], data['Category_item_cnt_M_sum'])
        return data

    def _logTransform(self, data):
        cols = data.columns.tolist()
        revenue_cols = []
        for col in cols:
            if 'revenue' in col:
                revenue_cols.append(col)
            else:
                pass

        for col in revenue_cols:
            data[col] = np.log1p(data[col])

        return data

    def transform(self, data):
        data = self._shopRevenue(data)
        data = self._itemRevenue(data)
        data = self._categoryRevenue(data)
        #         data = self._logTransform(data)

        return data

def trainTestSplit(data):
    '''Train and test data can be split by date_block_num.
    Tweleve training datasets stored in X_train and Y_train.'''

    X_train = []
    y_train = []
    for block_num in np.arange(22, 34, 1):
        X_train_block = data[data['date_block_num'] < block_num].iloc[:, 4:143]
        y_train_block = data[data['date_block_num'] < block_num].iloc[:, -1]

        X_train.append(X_train_block)
        y_train.append(y_train_block)

    X_val = data[data['date_block_num'] == 33].iloc[:, 4:143]
    y_val = data[data['date_block_num'] == 33].iloc[:, -1]

    test = data[data['date_block_num'] == 34].iloc[:, 4:143]

    return X_train, y_train, X_val, y_val, test

#catboost modeling
def rmse(y_actual, y_pred):
    '''Metrics - Root Mean Squared Error.
    y_actual: vector of actual values;
    y_pred: vector of test values.'''

    return np.sqrt(sklearn.metrics.mean_squared_error(y_actual, y_pred))

def catboostModel(X_train, Y_train, X_val, y_val, test):
    '''catboost modeling.'''
    scores = []
    pred = []
    for i in range(len(Y_train)):
        train_pool = Pool(X_train[i], Y_train[i])
        cbr = CatBoostRegressor(iterations=500, learning_rate=0.3, depth=5, l2_leaf_reg=10, loss_function='RMSE',
                                random_seed=0)
        cbr.fit(train_pool)
        y_hat = cbr.predict(X_val)
        y_pred = cbr.predict(test)
        scores.append(rmse(y_val, y_pred))
        pred.append(y_pred)

    return scores, pred

def submission(data):
    '''predicted values for submission.'''
    sub = pd.DataFrame(data={'ID': test.ID, 'item_cnt_month': data.mean()})
    sub.to_csv('sub_mean.csv', index=False)

    return sub

if __name__ == '__main__':
    sales_train, item_categories = trainDataLoadMerge() #Raw train data (daily) load.
    sales_train = preprocess(sales_train) #train data (daily) preprocess.

    train_m = aggbyMonth(sales_train) #Aggregate train data by month.
    # train_m = meanEncode(train_m, 100)

    train_m = aggItemsMonthly(sales_train, train_m, ['item_id', 'shop_id', 'category_id']).sort_values(['date_block_num', 'shop_id', 'item_id']) #aggregate more features by monthly

    test = testDefine(item_categories, train_m) #test data definitionn
    data_all_m = pd.concat([train_m, test.drop('ID', axis=1)]) #Add train and test for processing

    data_all_m = FeaturesTransform().transform(data_all_m) #process train and test data

    cols = data_all_m.columns.tolist()
    cols = cols[28:29] + cols[34:] + cols[31:32] + cols[27:28] + cols[32:34] + cols[7:9] + cols[16:18] + cols[25:27] + cols[29:31] + cols[0:7] + cols[9:16] + cols[18:25]
    data_all_m = data_all_m[cols] # re-organize columns for the coming creation of lag features.

    #data_all_m = Revenues().transform(data_all_m) #add revenues features

    data_all_m = lagFeatures(data_all_m) #create lag features for original cnt features.

    cols = data_all_m.columns.tolist()
    cols = cols[0:12] + cols[34:] + cols[13:34] + cols[12:13]
    data_all_m = data_all_m[cols] #re-organize columns for defining model-learnable train and test.

    X_train, y_train, X_val, y_val, test = trainTestSplit(data_all_m) #train and test data split and define for modeling;
    # twelve training datasets created for generating 12 predictions (and models). The final prediciton is the average of the 12 predictions.

    scores, pred = catboostModel(X_train, y_train, X_val, y_val, test)

    sub = submission(pred) #output to the pre-defined directory.

    print scores










