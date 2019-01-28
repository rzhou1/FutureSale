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

train_pool = Pool(X_train, y_train, cat_features=cat_features)

cbr = CatBoostRegressor(random_seed=0)

params = {'iterations': [100, 500, 1000, 2000], 'learning_rate': [0.03, 0.1, 0.3], 'depth': [3, 5,8], 'l2_leaf_reg': [3, 10], 'loss_function': ['RMSE']}
cbr = CatBoostRegressor()
gs = GridSearchCV(cbr, params, cv=3, n_jobs=4, verbose=1)
gs.fit(X_train, y_train, cat_features=cat_features)

y_hat = gs.predict(X_val_201510)

print rmse(Y_val_201510, y_hat)
print gs.best_params_


#Bayesian optimization by hyperopt: catboost

train_pool = Pool(X_train, y_train, cat_features=cat_features)

def objective(space):
    model = CatBoostRegressor(iterations=int(space['iterations']), l2_leaf_reg=space['l2_leaf_reg'], learning_rate=space['learning_rate'],
                              depth=int(space['depth']),
                              loss_function='RMSE')
    model.fit(train_pool)
    y_pred = model.predict(X_val_201510)

    score = rmse(y_val_201510, y_pred)

    return score


space = {'iterations': hp.choice('iterations', np.arange(100, 2100, 1000)),
         'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
         'depth': hp.quniform('depth', 3, 9, 2),
         'learning_rate': hp.logunifrom('learning_rate', np.log(0.01), np.log(0.3))}

trials = Trials()

result = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trials, max_evals=100, verbose=1)

print result
