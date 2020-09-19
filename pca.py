import pandas as pd

from sklearn.decomposition import PCA
from data_prep import load_X_Y, load_train, PCA_TRAIN, load_X_Y_file, DUMP_DIR

n_components = 4
file_name = 'no_ids'

X, Y = load_X_Y_file(file_name)

# print(X.columns)

pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(X)
print(f'pca.explained_variance_ = {pca.explained_variance_}')
print(f'pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}')

principalDf = pd.DataFrame(data = principalComponents
, columns = [f'pc {i}' for i in range(n_components)])

principalDf['damage_grade'] = Y

principalDf.to_csv(DUMP_DIR + file_name + f'_pca_{n_components}.csv')