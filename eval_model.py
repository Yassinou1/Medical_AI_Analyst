import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from services.openfda_service import MedicalDataService

# 1. Modell und Scaler laden
print("Lade medizinisches Modell und Scaler...")
model = load_model("models/lstm_model.keras")
scaler = joblib.load("models/scaler.pkl")
look_back = 24
drug_name = "ibuprofen"

# 2. Testdaten laden
print(f"Lade historische Daten für {drug_name}...")
service = MedicalDataService()
df = service.fetch_adverse_events_timeseries(drug_name)

# 3. Feature Engineering 
data = df[['Adverse_Events']].copy()
data['SMA_4'] = data['Adverse_Events'].rolling(window=4).mean()
data['SMA_12'] = data['Adverse_Events'].rolling(window=12).mean()
data['ROC'] = data['Adverse_Events'].pct_change()

data.replace([np.inf, -np.inf], np.nan, inplace=True)

data.dropna(inplace=True)

# 4. Daten skalieren und Sequenzen bilden
scaled_data = scaler.transform(data)

X_test, y_test_actual = [], []
dates = data.index[look_back:] 

for i in range(look_back, len(scaled_data)):
    X_test.append(scaled_data[i-look_back:i])
    y_test_actual.append(scaled_data[i, 0]) # nur die Adverse_Events (Spalte 0)

X_test = np.array(X_test)
y_test_actual = np.array(y_test_actual)

# 5. Vorhersagen berechnen
print("Berechne Vorhersagen...")
predicted_scaled = model.predict(X_test)

# 6. Rückwärtsskalierung (Inverse Transform)
dummy_pred = np.zeros((len(predicted_scaled), scaled_data.shape[1]))
dummy_pred[:, 0] = predicted_scaled[:, 0]
predicted_events = scaler.inverse_transform(dummy_pred)[:, 0]

dummy_actual = np.zeros((len(y_test_actual), scaled_data.shape[1]))
dummy_actual[:, 0] = y_test_actual
actual_events = scaler.inverse_transform(dummy_actual)[:, 0]

# Negative Vorhersagen (durch das Modell) auf 0 setzen, da es keine negativen Nebenwirkungen gibt
predicted_events = np.maximum(predicted_events, 0)

# 7. Fehler-Metriken berechnen
mae = mean_absolute_error(actual_events, predicted_events)
rmse = np.sqrt(mean_squared_error(actual_events, predicted_events))

print("-" * 30)
print(f"Modell Evaluierung für {drug_name}:")
print(f"Mean Absolute Error (MAE): {mae:.2f} Fälle")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Fälle")
print("-" * 30)

# 8. Den Plot zeichnen
plt.figure(figsize=(14, 7))
plt.plot(dates, actual_events, color='black', label=f'Tatsächliche Meldungen ({drug_name})', linewidth=2)
plt.plot(dates, predicted_events, color='red', label='LSTM Vorhersage', linestyle='dashed', alpha=0.8)

plt.title(f'KI Performance Backtesting: Pharmakovigilanz ({drug_name})', fontsize=16)
plt.xlabel('Datum', fontsize=12)
plt.ylabel('Gemeldete Nebenwirkungen (Wöchentlich)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()