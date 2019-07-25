import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import random
import pickle
import numpy as np
import datetime

lr = 1e-3
batch_size = 32
num_classes = 2
eps = 2000

input_dim = [60, 30, 60]
data_file = '../data/pp-07231305.pkl'


class Simple_Net():
    def __init__(self, load=None):
        self.model = None
        self.class_weight = None
        self.model_construct()
        self.model_compile()
        if load:
            self.load_checkpoint(load)
        self.timestamp = datetime.datetime.now().strftime('%m%d%H%M')
        self.save = f'ckpts/{self.timestamp}'

    def model_construct(self):
        model_input = Input(shape=input_dim)
        conv1 = Conv2D(16, 3, activation='relu')(model_input)
        pool1 = MaxPooling2D(2)(conv1)
        conv2 = Conv2D(32, 2, activation='relu')(pool1)
        pool2 = MaxPooling2D(2)(conv2)
        conv3 = Conv2D(32, 2, activation='relu')(pool2)
        pool3 = MaxPooling2D(2)(conv3)
        flat = Flatten()(pool3)
        fc1 = Dense(256, activation='relu')(flat)
        fc2 = Dense(128, activation='relu')(fc1)
        out = Dense(num_classes, activation='softmax')(fc2)
        self.model = Model(inputs=model_input, outputs=out)
        
    def model_compile(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(lr),
            metrics=['accuracy'])
        return

    def model_fit(self, x, label, batch_size=32, epochs=8):
        history = self.model.fit(
            x, label,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            shuffle=True,
            class_weight=self.class_weight,
            validation_split=0.15,
        )
        return history

    def model_eval(self, x, label, batch_size=32):
        history = self.model.evaluate(
            x, label,
            batch_size=batch_size,
            verbose=0,
        )
        return history

    def save_checkpoint(self):
        self.model.save_weights(self.save)
        print(f"Saved model at {self.save}.")

    def load_checkpoint(self, load):
        self.model.load_weights(f'./network/ckpts/{load}')
        print(f"Loaded model from {load}.")

    def model_predict(self, data):
        label = self.model.predict(data)
        label = np.argmax(label, axis=1)
        return label


def read_file(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data

def prepare_data(data_file):
    data = read_file(data_file)
    n_cls = len(data.keys())
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    train_lens = []
    test_lens = []

    # data_len = np.min([objs.shape[0] for key, objs in data.items()])
    # train_len = int(data_len * 0.85)
    # test_len = data_len - train_len

    for i, (key, objs) in enumerate(data.items()):
        np.random.shuffle(objs)
        data_len = objs.shape[0]

        data_len = objs.shape[0]
        train_len = int(data_len * 0.85)
        test_len = data_len - train_len
        
        x_train.append(objs[:train_len])
        y_train += ([i] * train_len)
        x_test.append(objs[train_len:train_len+test_len])
        y_test += ([i] * test_len)
        train_lens.append(train_len)
        test_lens.append(test_len)

    x_train = np.concatenate(x_train)
    y_train = keras.utils.to_categorical(np.stack(y_train))
    x_test = np.concatenate(x_test)
    y_test = keras.utils.to_categorical(np.stack(y_test))

    print(f'{np.sum(train_lens)} Train Data:', train_lens)
    print(f'{np.sum(test_lens)} Test Data:', test_lens)

    return n_cls, x_train, y_train, x_test, y_test

def main():
    net = Simple_Net()
    n_cls, x_train, y_train, x_test, y_test = prepare_data(data_file)

    best_acc = 0
    for i in range(eps):
        his_train = net.model_fit(x_train, y_train).history
        his_test = net.model_eval(x_test, y_test)
        train_acc = np.average(his_train['acc'])
        train_loss = np.average(his_train['loss'])
        test_acc = np.average(his_test[1])
        test_loss = np.average(his_test[0])
        print(f'Eps {i}, Train acc {train_acc:.3f}, loss {train_loss:.3f}, Test acc {test_acc:.3f}, loss {test_loss:.3f}')
        if test_acc > best_acc:
            net.save_checkpoint()
        best_acc = max(best_acc, test_acc)

if __name__ == '__main__':
    main()

