import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
%matplotlib inline

sales_train = pd.read_csv('sales_train.csv')
sales_train.iloc[:, 0] = pd.to_datetime(sales_train.iloc[:, 0], format="%d.%m.%Y")

holidays = pd.read_feather('hol.feather')
item_en = pd.read_feather('it_en.feather')
shops_en = pd.read_feather('sh_en.feather')
item_categories_en = pd.read_feather('itc_en.feather')
holidays = holidays.rename(columns={'h_name': 'holiday_name', 'h_type': 'holiday_type'})
holidays['Is_Holiday'] = pd.Series('Yes', index=holidays.index)
item_en = item_en.rename(columns={'it_name': 'item_name', 'it_id': 'item_id', 'itc_id': 'item_category_id',
                        'en_it_name': 'item_name_en'})
shops_en = shops_en.rename(columns={'sname': 'shop_name', 'sid': 'shop_id', 'en_sname': 'shop_name_en'})
item_categories_en = item_categories_en.rename(columns={'itc_name': 'item_category_name', 'itc_id': 'item_category_id',
                                                        'en_itc_name': 'item_category'})

train = pd.merge(sales_train, item_en, on='item_id', how='left')
train = pd.merge(train, shops_en, on='shop_id', how='left')
train = pd.merge(train, item_categories_en, on='item_category_id', how='left')
train = pd.merge(train, holidays, on='date', how='left')

train['year'] = pd.DatetimeIndex(train['date']).year
train['month'] = pd.DatetimeIndex(train['date']).month
train['day'] = pd.DatetimeIndex(train['date']).day