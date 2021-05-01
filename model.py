from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from tensorflow import optimizers
from tensorflow.python.keras.layers import Dense, Dropout, Normalization
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

tf.compat.v1.disable_eager_execution()


def load_dataset(data_folder_csv):
    # load the dataset as a pandas DataFrame
    data = pd.read_csv(data_folder_csv, header=0)
    # retrieve numpy array
    dataset = data.values

    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:, -1]
    print(y)

    # format all fields as floats
    X = X.astype(np.float)
    # reshape the output variable to be one column (e.g. a 2D shape)
    y = y.reshape((len(y), 1))
    return X, y


# prepare input data using min/max scaler.
def prepare_inputs(X_train, X_test):
    oe = MinMaxScaler()
    X_train_enc = oe.fit_transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    ohe = OneHotEncoder()
    y_train = le.fit_transform(y_train).reshape(-1, 1)
    y_test = le.transform(y_test).reshape(-1, 1)
    y_train_enc = ohe.fit_transform(y_train).toarray()
    y_test_enc = ohe.transform(y_test).toarray()
    return y_train_enc, y_test_enc


df = pd.read_csv("csv_ready1.csv")

df = df.fillna(value=0.5)
df.to_csv("csv_ready1.csv")
X, y = load_dataset("csv_ready1.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)

# prepare_input function missing here
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
print('Finished preparing inputs.')

# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

model = Sequential()
model.add(Dense(64, input_shape=X_train.shape[1:], activation="tanh", kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train_enc, y_train_enc, epochs=20, batch_size=128, verbose=1, use_multiprocessing=True)

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
