
from keras.layers import Dense, Input

from data_prep import load_X_Y


class KerasClassifier:
    def __init__(self, hidden_layers, df):
        X, Y = load_X_Y(df)

        input_tensor_size = len(X.columns)
        output_size = 5

