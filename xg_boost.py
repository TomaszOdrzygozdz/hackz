from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np

from data_prep import load_X_Y_file

MODEL_FILE = 'saved_models/xgb_v0'

test_size = 0.1

X, Y = load_X_Y_file('no_ids_pca_4')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

model = XGBClassifier(objective='multi:softmax')
# model.fit(X, Y)
model.load_model(MODEL_FILE)
#model.save_model(MODEL_FILE)

y_pred = model.predict(X_test)
output_shape = y_pred.shape
y_test_array = (y_test.values).reshape(output_shape)
correct = [y_test_array[i] == y_pred[i] for i in range(len(y_pred))]

def RMSEcalc(a, b):
    return np.sqrt(metrics.mean_squared_error(a, b))
print('Accuracy = {}')

