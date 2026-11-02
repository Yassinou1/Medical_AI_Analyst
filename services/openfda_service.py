"""
OpenFDA API Service for fetching adverse events timeseries data.
"""

import datetime
import logging
from typing import Dict, Any

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings

logger = logging.getLogger(__name__)

__all__ = ['MedicalDataService', 'OpenFDAAPIError']

class OpenFDAAPIError(Exception):
    """Custom exception raised for OpenFDA API failures."""
    pass

class MedicalDataService:
    """Service to interact with the OpenFDA API to retrieve medical data."""
    
    def __init__(self) -> None:
        """Initialize the MedicalDataService."""
        self.base_url = settings.OPENFDA_BASE_URL
        self.start_date = settings.OPENFDA_DATA_START_DATE

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError, OpenFDAAPIError)),
        reraise=True
    )
    async def fetch_adverse_events_timeseries(self, drug_name: str) -> pd.DataFrame:
        """
        Fetch adverse events timeseries data for a given drug asynchronously.
        
        Args:
            drug_name (str): The name of the drug.
            
        Returns:
            pd.DataFrame: A DataFrame containing the weekly adverse events.
            
        Raises:
            OpenFDAAPIError: If the API request fails after retries.
        """
        logger.info(f"Lade Nebenwirkungs-Daten (ab {self.start_date}) für '{drug_name}'...")
        
        today_str = datetime.datetime.now().strftime("%Y%m%d")
        
        params = {
            "search": f'(patient.drug.medicinalproduct:"{drug_name}") AND (receiptdate:[{self.start_date} TO {today_str}])',
            "count": "receiptdate"
        }
        
        async with httpx.AsyncClient(timeout=settings.OPENFDA_REQUEST_TIMEOUT) as client:
            try:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                df = pd.DataFrame(data['results'])
                if df.empty:
                    return pd.DataFrame(columns=['Adverse_Events'])
                
                df['time'] = pd.to_datetime(df['time'], format='%Y%m%d')
                df.set_index('time', inplace=True)
                df.rename(columns={'count': 'Adverse_Events'}, inplace=True)
                
                weekly_df = df.resample('W').sum()
                
                return weekly_df
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error occurred: {e}")
                raise OpenFDAAPIError(f"Fehler beim Abruf der FDA API: {e}") from e
            except httpx.RequestError as e:
                logger.error(f"Request error occurred: {e}")
                raise OpenFDAAPIError(f"Verbindungsfehler zur FDA API: {e}") from e