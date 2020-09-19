import re

import neptune
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor

from numpy import mean
import random
from data_prep import load_train, target, load_test, ids_columns, save_final_output, balance_dataset, dump_predictions
from neptune_helper import NeptuneHelper
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from boruta import BorutaPy
from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel, f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import lightgbm as lightgbm

# import lightgbm
import numpy as np
# import xgboost
from sklearn.ensemble import  RandomForestRegressor, RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, ElasticNet, BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

nh = NeptuneHelper('test1')


def scale_and_select_features(features, y):

    def cor_selector(X, y):
        feature_names = X.columns.tolist()
        cor_list = []
        for i in X.columns.tolist():
            cor = np.corrcoef(X[i], y)[0, 1]
            cor_list.append(cor)
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-100:]].columns.tolist()
        cor_support = [True if i in cor_feature else False for i in feature_names]
        return cor_support

    def _f_regr(X, y):
        X_norm = MinMaxScaler().fit_transform(X)
        chi_selector = SelectKBest(f_regression, k='all')
        chi_selector.fit(X_norm, y)
        chi_support = chi_selector.get_support()
        return chi_support

    def rfe(X, y):
        X_norm = MinMaxScaler().fit_transform(X)
        rfe_selector = RFE(estimator=RandomForestRegressor(n_estimators=5), step=10, verbose=5)
        rfe_selector.fit(X_norm, y)
        rfe_support = rfe_selector.get_support()
        return rfe_support

    def embedded_rf(X, y):
        embeded_rf_selector = SelectFromModel(RandomForestRegressor(n_estimators=5), threshold='1.25*median')
        embeded_rf_selector.fit(X, y)
        embeded_rf_support = embeded_rf_selector.get_support()
        return embeded_rf_support

    def embedded_lgbm(X, y):
        lgbc = lightgbm.LGBMRegressor(objective='regression', num_leaves=5,
                                        learning_rate=0.05, n_estimators=5,
                                        max_bin=55, bagging_fraction=0.8,
                                        bagging_freq=5, feature_fraction=0.2319,
                                        feature_fraction_seed=9, bagging_seed=9,
                                        min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
        embeded_lgb_selector = SelectFromModel(lgbc, threshold='1.25*median')
        embeded_lgb_selector.fit(X, y)
        embeded_lgb_support = embeded_lgb_selector.get_support()
        return embeded_lgb_support

    def extra_trees(X, y):
        model = ExtraTreesRegressor(n_estimators=5)
        model.fit(X, y)
        support = np.array(model.feature_importances_ > 0.01)
        return support

    def boruta(X, y):
        X_norm = MinMaxScaler().fit_transform(X)
        rf = RandomForestRegressor(n_estimators=5)
        feat_selector = BorutaPy(rf, n_estimators=5, verbose=2)
        feat_selector.fit_transform(X_norm, y)
        return feat_selector.support_

    def select_top_mostly_selected_features():
        supports = [fn(features, y) for fn in [cor_selector, _f_regr, rfe, embedded_rf, embedded_lgbm, extra_trees, boruta]]

        feature_selection_df = pd.DataFrame(
            {'Feature': features.columns.tolist(), 'Pearson': supports[0],  'F_regr': supports[1],
             'RFE': supports[2], 'Random Forest': supports[3], 'LightGBM': supports[4], 'ExtraTrees': supports[5], 'Boruta': supports[6]})

        feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
        # print(feature_selection_df)

        max_total = math.ceil(len(supports)*0.5) + 1
        feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
        feature_selection_df.index = range(1, len(feature_selection_df) + 1)
        feature_names = feature_selection_df[feature_selection_df['Total'] >= max_total]['Feature'].to_list()
        return feature_names

    if features.shape[0] > 0:
        feature_names = select_top_mostly_selected_features()
        features = features[feature_names]
        features = pd.DataFrame(features, columns=features.columns)
    return features, y


def return_inner_models(features):

    models = []

    ### random forest model
    rf_model = RandomForestClassifier(max_depth=5, random_state=42, n_estimators=5,
                                     verbose=0)  # Train the model on training data
    models.append(("rf", rf_model))

    gnb = GaussianNB()
    models.append(('gnb', gnb))
    # #
    # # ### ridge
    ridge = RidgeClassifier()
    models.append(('ridge', ridge))
    # #
    # # ### mlp
    # mlp = MLPClassifier(alpha=1, max_iter=100)
    # models.append(("mlp", mlp))
    # #
    # # ### knn
    knn = KNeighborsClassifier(n_neighbors=3)
    models.append(("knn", knn))
    # #
    # # ### adaboost
    ada = AdaBoostClassifier(n_estimators=5, random_state=0)
    models.append(("ada", ada))
    #
    # ### decision treee
    dtree = DecisionTreeClassifier(max_depth=5)
    models.append(("dtree", dtree))
    #
    # ### lgbm model
    lgbm_model = lightgbm.LGBMClassifier(num_leaves=5,
                                        learning_rate=0.05, n_estimators=5,
                                        max_bin=55, bagging_fraction=0.8,
                                        bagging_freq=5, feature_fraction=0.2319,
                                        feature_fraction_seed=9, bagging_seed=9,
                                        min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
    models.append(("lgbm",lgbm_model))
    return models


def TrainSKClassificator():

    # selected_dataset = load_dbalanced_train_df  #### nonans_dataset#noinfo_dataset #### tutaj usuwamy te rzedzy co maja duzo kolumn, ktore sa puste
    # selected_dataset
    X = load_train()
    # assert X[X.isnull().any(axis=1)].shape[0] > 0
    # X.shape
    X = balance_dataset(X)
    X_test = load_test()
    # assert X[X.isnull().any(axis=1)].shape[0] > 0
    # X_test[X_test.isnull().any(axis=1)]

    X_test_id = X_test['building_id']
    # X.drop(columns=ids_columns, inplace=True)
    # X_test.drop(columns=ids_columns, inplace=True)
    print(X_test_id)
    X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    X_test = X_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    X.reset_index(drop=True, inplace=True)
    X_to_select = X.sample(n=int(X.shape[0]/5))
    X_to_sel, y_to_sel = X_to_select.loc[:, X_to_select.columns != target], X_to_select[[target]]

    X, Y = X.loc[:, X.columns != target], X[[target]]
    # y[y.isnull().any(axis=1)]
    # X_norm = MinMaxScaler().fit_transform(X)
    # rf = RandomForestRegressor(n_estimators=5)
    # feat_selector = BorutaPy(rf, n_estimators=5, verbose=2)
    # feat_selector.fit_transform(X_norm, y)
    # feat_selector.support_


    # X_to_select = X_to_select.sample(n=X_to_select.shape[0]/10)
    a, b = scale_and_select_features(X_to_sel, y_to_sel.iloc[:,0])
    # selected_dataset
    # y
    # X, y = a,b
    selected_cols = a.columns.copy()
    print(selected_cols)

    X = X[selected_cols]
    X_test = X_test[selected_cols]

    # X = load_train()
    # X, Y = X.loc[:, X.columns != target], X[[target]]

    def RMSEcalc(a, b):
        return np.sqrt(metrics.mean_squared_error(a, b))


    from sklearn.model_selection import KFold

    SPLITS = 10
    for i in range(1):
        kf = KFold(n_splits=SPLITS, random_state=random.randint(0, 2**32-1), shuffle=True)

        model = VotingClassifier(return_inner_models(X))#n_estimators=5, max_depth=5)
        RMSE_sum = 0
        RMSE_length = SPLITS
        r = []

        print('train models')
        print(X.shape)
        for loop_number, (train, test) in enumerate(kf.split(X)):

            training_X_array = X.reindex(index=train)
            training_y_array = Y.reindex(index=train)

            X_test_array = X.reindex(index=test)
            y_actual_values = Y.reindex(index=test)
            print('model fitting')
            model.fit(training_X_array, training_y_array)
            print('model fitted')
            print('predicting')
            print(X_test_array.shape)
            prediction = model.predict(X_test_array)
            print('predicted')

            crime_probabilites = np.array(prediction)
            print('calculating rmse')
            RMSE_cross_fold = RMSEcalc(crime_probabilites, y_actual_values)
            print('rmse calculated')
            r.append(RMSE_cross_fold)
            nh.log_metric('rmse_one_fold', RMSE_cross_fold)

            RMSE_sum=RMSE_cross_fold+RMSE_sum

        print('predict')
        output_ = model.predict(X_test)
        dump_predictions(X_test_id, output_)

        print('saved')

        RMSE_cross_fold_avg=RMSE_sum/RMSE_length
        nh.log_metric('RMSE avergae', RMSE_sum/RMSE_length)
        print('The Mean RMSE across all folds is',RMSE_cross_fold_avg)
        print(r)


TrainSKClassificator()