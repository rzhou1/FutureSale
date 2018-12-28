import numpy as np
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

#metrics definition
def rmse(y_actual, y_pred):
    '''Metrics - Root Mean Squared Error.
    y_actual: vector of actual values;
    y_pred: vector of test values.'''

    return np.sqrt(mean_squared_error(y_actual, y_pred))

#GridSearchCV

train_pool = Pool(X_train_201409, Y_train_201409)

cbr = CatBoostRegressor(random_seed=0)
cbr.fit(train_pool)
params = {'iterations': [100, 500, 1000, 2000], 'learning_rate': [0.03, 0.1, 0.3], 'depth': [3, 5,8], 'l2_leaf_reg': [3, 10], 'loss_function': ['RMSE']}
cbr = CatBoostRegressor()
gs = GridSearchCV(cbr, params, cv=3, n_jobs=4, verbose=1)
gs.fit(X_train_201509, Y_train_201509)

y_hat = gs.predict(X_val_201510)

print rmse(Y_val_201510, y_hat)
print gs.best_params_


#Bayesian optimization by hyperopt: catboost

train_pool = Pool(X_train_201409, Y_train_201409)

def objective(space):
    model = CatBoostRegressor(iterations=int(space['iterations']), l2_leaf_reg=space['l2_leaf_reg'], learning_rate=0.03,
                              depth=int(space['depth']),
                              loss_function='RMSE')
    model.fit(train_pool)
    y_pred = model.predict(X_val_201510)

    score = rmse(Y_val_201510, y_pred)

    return score


space = {'iterations': hp.choice('iterations', np.arange(100, 2100, 1000)),
         'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
         'depth': hp.quniform('depth', 3, 9, 2)}

trials = Trials()

result = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trials, max_evals=100, verbose=1)

print result

#Bayesian optimization by hyperopt: xgboost

def objective(space):
    model = xgb.XGBRegressor(n_estimators=int(space['n_estimators']), learning_rate=0.03,
                             max_depth=int(space['max_depth']), min_child_weight=int(space['min_child_weight']),
                             subsample=space['subsample'], colsample_bytree=space['colsample_bytree'],
                             gamma=space['gamma'], reg_lambda=space['reg_lambda'])
    model.fit(X_train_201409, Y_train_201409)
    y_pred = model.predict(X_val_201510)

    score = rmse(Y_val_201510, y_pred)

    return score


space = {'n_estimators': hp.quniform('n_estimators', 100, 1100, 500),
         'max_depth': hp.quniform('max_depth', 3, 11, 2),
         'min_child_weight': hp.quniform('min_child_weight', 2, 8, 1),
         'subsample': hp.uniform('subsample', 0.7, 1),
         'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1),
         'gamma': hp.uniform('gamma', 0.1, 0.5),
         'reg_lambda': hp.uniform('reg_lambda', 0, 1)}

trials = Trials()

best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials, verbose=1)

print best_params