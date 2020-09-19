import pandas as pd

from sklearn.decomposition import PCA
from data_prep import load_X_Y, load_train, PCA_TRAIN

n_components = 5


X, Y = load_X_Y(load_train())

pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
, columns = [f'pc {i}' for i in range(n_components)])

principalDf['damage_grade'] = Y

principalDf.to_csv(PCA_TRAIN + f'{n_components}.csv')