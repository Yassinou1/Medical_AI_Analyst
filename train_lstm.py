import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from services.openfda_service import MedicalDataService

os.makedirs("models", exist_ok=True)

drug_name = "ibuprofen"
service = MedicalDataService()
df = service.fetch_adverse_events_timeseries(drug_name)

data = df[['Adverse_Events']].copy()

# Feature Engineering für medizinische Trendanalyse
data['SMA_4'] = data['Adverse_Events'].rolling(window=4).mean()
data['SMA_12'] = data['Adverse_Events'].rolling(window=12).mean()
data['ROC'] = data['Adverse_Events'].pct_change()

data.replace([np.inf, -np.inf], np.nan, inplace=True)

data.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

look_back = 24
num_features = scaled_data.shape[1] 

x_train, y_train = [], []
for i in range(look_back, len(scaled_data)):
    x_train.append(scaled_data[i-look_back:i])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(look_back, num_features)))
model.add(Dropout(0.2))

model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=16, epochs=15)

model.save("models/lstm_model.keras")
joblib.dump(scaler, "models/scaler.pkl")