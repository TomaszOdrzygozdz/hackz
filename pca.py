import pandas as pd

from sklearn.decomposition import PCA
from data_prep import load_X_Y, load_train, PCA_TRAIN, load_X_Y_file, DUMP_DIR, \
    load_test

n_components = 100
file_name = 'train_simple_new'
test_file_name = 'test_simple_new'

X, Y = load_X_Y_file(file_name)
X_test = load_test(test_file_name)
assert X_test[X_test.isnull().any(axis=1)].shape[0] == 0
# print(X.columns)

pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(X)
principalComponentsTest = pca.transform(X_test)
print(f'pca.explained_variance_ = {pca.explained_variance_}')
print(f'pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}')
print(f'pca.explained_variance_sum = {sum(pca.explained_variance_ratio_)}')

principalDf = pd.DataFrame(data = principalComponents
, columns = [f'pc {i}' for i in range(n_components)])

principalTestDf = pd.DataFrame(data = principalComponentsTest
, columns = [f'pc {i}' for i in range(n_components)])

principalDf['damage_grade'] = Y
principalDf.to_csv(DUMP_DIR + file_name + f'_pca_{n_components}.csv', index=False)
principalTestDf.to_csv(DUMP_DIR + test_file_name + f'_pca_{n_components}.csv', index=False)