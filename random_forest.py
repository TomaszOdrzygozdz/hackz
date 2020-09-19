import neptune
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from numpy import mean
import random
from data_prep import load_train, target
from neptune_helper import NeptuneHelper

nh = NeptuneHelper('test1')

def TrainSKClassificator():

    X = load_train()
    X, Y = X.loc[:, X.columns != target], X[[target]]

    def RMSEcalc(a, b):
        return np.sqrt(metrics.mean_squared_error(a, b))
    from sklearn.model_selection import KFold

    for i in range(1):
        kf = KFold(n_splits=10, random_state=random.randint(0, 2**32-1), shuffle=True)

        model = RandomForestClassifier(n_estimators=5, max_depth=5)
        RMSE_sum=0
        RMSE_length=10
        r = []

        for loop_number, (train, test) in enumerate(kf.split(X)):

            training_X_array = X.reindex(index=train)
            training_y_array = Y.reindex(index=train)

            X_test_array = X.reindex(index=test)
            y_actual_values = Y.reindex(index=test)
            model.fit(training_X_array, training_y_array)
            prediction = model.predict(X_test_array)
            crime_probabilites = np.array(prediction)
            RMSE_cross_fold = RMSEcalc(crime_probabilites, y_actual_values)
            r.append(RMSE_cross_fold)
            nh.log_metric('rmse_one_fold', RMSE_cross_fold)

            RMSE_sum=RMSE_cross_fold+RMSE_sum

        RMSE_cross_fold_avg=RMSE_sum/RMSE_length
        nh.log_metric('RMSE avergae', RMSE_sum/RMSE_length)
        print('The Mean RMSE across all folds is',RMSE_cross_fold_avg)
        print(r)


TrainSKClassificator()