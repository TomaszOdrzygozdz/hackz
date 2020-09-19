from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from data_prep import load_X_Y_file

test_size = 0.1

X, Y = load_X_Y_file('no_ids_pca_4')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

model = XGBClassifier(objective='multi:softmax')
#model.fit(X, Y)
#model.save_model('saved_models/xgb_v0')

y_pred = model.predict(X_test)
predictions = [np.argmax(value) for value in y_pred]


