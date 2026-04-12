import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import sys
import os

# Pfad-Fix, falls der Service aus main.py im Root aufgerufen wird
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.openfda_service import MedicalDataService

class LSTMService:
    def __init__(self):
        self.model = load_model("models/lstm_model.keras")
        self.scaler = joblib.load("models/scaler.pkl")
        self.look_back = 24
        self.data_service = MedicalDataService()

    def predict_next_week(self, drug_name: str) -> int:
        df = self.data_service.fetch_adverse_events_timeseries(drug_name)
        
        data = df[['Adverse_Events']].copy()
        data['SMA_4'] = data['Adverse_Events'].rolling(window=4).mean()
        data['SMA_12'] = data['Adverse_Events'].rolling(window=12).mean()
        data['ROC'] = data['Adverse_Events'].pct_change()

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        data.dropna(inplace=True)
        recent_data = data.tail(self.look_back)
        
        scaled_data = self.scaler.transform(recent_data)
        X_test = np.reshape(scaled_data, (1, self.look_back, scaled_data.shape[1]))
        
        predicted_scaled = self.model.predict(X_test)
        
        dummy_array = np.zeros((1, scaled_data.shape[1]))
        dummy_array[0, 0] = predicted_scaled[0][0] 
        
        predicted_events = self.scaler.inverse_transform(dummy_array)[0, 0]
        
        return int(max(0, predicted_events))