from pathlib import Path

import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from tensorflow import optimizers
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizer_v2.adam import Adam


def load_dataset(data_folder_csv):
    # load the dataset as a pandas DataFrame
    data = pd.read_csv(data_folder_csv, header=None)
    # retrieve numpy array
    dataset = data.values

    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:, -1]
    print(y)

    # format all fields as string
    X = X.astype(str)
    # reshape the output variable to be one column (e.g. a 2D shape)
    y = y.reshape((len(y), 1))
    return X, y


# prepare input data using min/max scaler.
def prepare_inputs(X_train, X_test):
    oe = RobustScaler().fit_transform(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


# prepare input data using 1-HOT
def prepare_inputs_ohot(X_train, X_test):
    ohe = OneHotEncoder(handle_unknown = 'ignore')
    ohe.fit(X_train)
    X_train_enc = ohe.transform(X_train)
    X_test_enc = ohe.transform(X_test)
    return X_train_enc, X_test_enc


# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc


# load the dataset
X, y = load_dataset("csv_ready.csv")
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)

X_train_enc, X_test_enc = prepare_inputs_ohot(X_train, X_test)
print('Finished preparing inputs.')
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
print('Finished preparing outputs.')
# define the  model
model = Sequential()
model.add(Dense(187, input_dim=X_train_enc.shape[1], activation="tanh", kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(64, input_dim=X_train_enc.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, input_dim=X_train_enc.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train_enc, y_train_enc, epochs=20, batch_size=128, verbose=1, use_multiprocessing=True)
# evaluate the keras model
_, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
print('Accuracy: %.2f' % (accuracy * 100))



'''
Accuracy: 49.95
model = Sequential()
model.add(Dense(32, input_dim=X_train_enc.shape[1], activation="tanh", kernel_initializer='he_normal'))
model.add(Dropout(0.4))
model.add(Dense(16, input_dim=X_train_enc.shape[1], activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(8, input_dim=X_train_enc.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

'''