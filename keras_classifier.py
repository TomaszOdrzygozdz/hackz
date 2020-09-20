
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.utils import np_utils
from data_prep import load_X_Y, load_train, load_X_Y_file
from neptunecontrib.monitoring.keras import NeptuneMonitor

from neptune_helper import NeptuneHelper


class KerasClassifier:
    def __init__(self, hidden_layers, file_name):

        nh = NeptuneHelper('KerasClassifier' ,params = {'hidden_layers': hidden_layers})
        self.X, self.Y = load_X_Y_file(file_name)

        input_tensor_size = len(self.X.columns)
        output_size = 6

        self.model = Sequential()
        self.model.add(Dense(hidden_layers.pop(0), input_dim=input_tensor_size, activation='relu'))
        for hidden_layer in hidden_layers:
            self.model.add(Dense(hidden_layer, activation='relu'))
        self.model.add(Dense(output_size, activation='softmax'))
        # Compile model
        self.model.compile(loss='mse', optimizer='adam',
                      metrics=['accuracy'])

    def train(self, epochs=5):
        X_np = self.X.values
        Y_np = np_utils.to_categorical(self.Y, num_classes=6)
        Y_np_sums = Y_np.sum(axis=0)
        print(Y_np_sums)
        self.model.fit(x=X_np, y=Y_np, epochs=epochs, callbacks=[NeptuneMonitor()], validation_split=0.1)
        print('Training done')
        yp = self.model.predict(x=X_np)
        print(yp[0])

kc = KerasClassifier([300,300, 300,300], 'train_simple_pca_75')
kc.train(epochs=10)
