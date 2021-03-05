import os
import tempfile
from pathlib import Path
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

data_folder_csv = Path("csv_ready.csv")
dataset = pd.read_csv(data_folder_csv, encoding='utf-8')
df = pd.DataFrame(dataset)  # dataframe for easier handling of the data.


# create a scaler object, this object will scale all values between 0-1
scaler = MinMaxScaler()
# fit and transform the data
df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print(df_norm)

train, test = train_test_split(df_norm, test_size=0.7)
train, val = train_test_split(train, test_size=0.7)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
	dataframe = dataframe.copy()
	labels = dataframe.pop('koi_score')
	ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
	if shuffle:
		ds = ds.shuffle(buffer_size=len(dataframe))
	ds = ds.batch(batch_size)
	return ds


batch_size = 16
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# a path for the model.
model_dir = tempfile.gettempdir()
model_ver = 1.0
export_path = os.path.join(model_dir, str(model_ver))

# will create a file checkpoint for our model, it will overwrite it every run until we will find the best model
checkpoint = ModelCheckpoint(filepath=export_path + '\model',
                             monitor='val_loss',  # monitor our progress by loss value.
                             mode='min',  # smaller loss is better, we try to minimize it.
                             save_best_only=True,
                             verbose=1)

# if our model accuracy (loss) is not improving over 3 epochs, stop the training, something is fishy
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

# if our loss is not improving, try to reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [checkpoint, earlystop, reduce_lr]

feature_columns = []

# numeric cols
for header in ['koi_disposition', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_period',
               'koi_time0bk',
               'koi_time0', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_ror', 'koi_srho', 'koi_prad', 'koi_sma',
               'koi_incl', 'koi_teq', 'koi_insol', 'koi_dor', 'koi_model_snr', 'koi_count', 'koi_num_transits',
               'koi_tce_plnt_num', 'koi_bin_oedp_sig', 'koi_steff', 'koi_slogg', 'koi_smet', 'koi_srad',
               'koi_smass', 'koi_kepmag', 'koi_gmag', 'koi_rmag', 'koi_imag', 'koi_zmag', 'koi_jmag', 'koi_hmag',
               'koi_kmag', 'koi_fwm_stat_sig', 'koi_fwm_sra', 'koi_fwm_sdec', 'koi_fwm_srao', 'koi_fwm_sdeco',
               'koi_fwm_prao', 'koi_fwm_pdeco', 'koi_dicco_mra', 'koi_dicco_mdec', 'koi_dicco_msky', 'koi_dikco_mra',
               'koi_dikco_mdec', 'koi_dikco_msky']:
	feature_columns.append(tf.feature_column.numeric_column(header))

# our feature layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
	feature_layer,
	layers.Dense(128, activation='relu'),
	layers.Dense(128, activation='relu'),
	layers.Dropout(.1),
	layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              callbacks=callbacks)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=50)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
