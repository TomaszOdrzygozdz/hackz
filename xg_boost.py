from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from data_prep import load_X_Y_file, DATA_DIR, dump_predictions, load_test

file_name = 'train_simple_new_pca_100'
file_name_test = 'test_simple_new_pca_100'

MODEL_FILE = 'saved_models/' + file_name

test_size = 0.01

X, Y = load_X_Y_file(file_name)
original_X_test = load_test(file_name_test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

X_test_building_id = pd.read_csv(DATA_DIR + 'test.csv')['building_id']

model = XGBClassifier(objective='multi:softmax')
model.fit(X, Y)
#model.load_model(MODEL_FILE)
model.save_model(MODEL_FILE)

y_pred = model.predict(X_test)

real_test_set_pred = model.predict(original_X_test)
dump_predictions(X_test_building_id, real_test_set_pred)
output_shape = y_pred.shape
y_test_array = (y_test.values).reshape(output_shape)
correct = [y_test_array[i] == y_pred[i] for i in range(len(y_pred))]

def RMSEcalc(a, b):
    return np.sqrt(metrics.mean_squared_error(a, b))
print(f'Accuracy = {np.mean(correct)}')
print(f'RMSE = {RMSEcalc(y_pred, y_test_array)}')

