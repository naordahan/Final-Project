from pathlib import Path
import pandas as pd
from keras_applications.densenet import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data_folder_csv = Path("csv_ready.csv")
dataset = pd.read_csv(data_folder_csv, encoding='utf-8')
df = pd.DataFrame(dataset)  # dataframe for easier handling of the data.

train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# create a scaler object, this object will scale all values between 0-1
scaler = MinMaxScaler()
# fit and transform the data
df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print(df_norm)
