import requests
import pandas as pd
import datetime 

class MedicalDataService:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/event.json"

    def fetch_adverse_events_timeseries(self, drug_name: str) -> pd.DataFrame:
        print(f"Lade Nebenwirkungs-Daten (ab 2020) für '{drug_name}'...")
        
        today_str = datetime.datetime.now().strftime("%Y%m%d")
        
        params = {
            "search": f'(patient.drug.medicinalproduct:"{drug_name}") AND (receiptdate:[20200101 TO {today_str}])',
            "count": "receiptdate"
        }
        
        response = requests.get(self.base_url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Fehler beim Abruf der FDA API: {response.text}")
            
        data = response.json()
        
        df = pd.DataFrame(data['results'])
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d')
        df.set_index('time', inplace=True)
        df.rename(columns={'count': 'Adverse_Events'}, inplace=True)
        
        weekly_df = df.resample('W').sum()
        
        return weekly_df