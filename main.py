from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.lstm_service import LSTMService
from services.agent_service import AgentService 

app = FastAPI(title="Medical AI Analyst API")

lstm_service = None
agent_service = None 

@app.on_event("startup")
def load_services():
    global lstm_service, agent_service
    lstm_service = LSTMService()
    agent_service = AgentService() 

class MedicalRequest(BaseModel):
    drug_name: str

@app.post("/analyze")
def analyze_drug(request: MedicalRequest):
    try:
        predicted_events = lstm_service.predict_next_week(request.drug_name)
        agent_summary = agent_service.analyze_with_agent(request.drug_name, predicted_events)
        
        return {
            "drug_name": request.drug_name.lower(),
            "quantitative_analysis": {
                "model": "LSTM (Lookback 24 Weeks)",
                "predicted_adverse_events": predicted_events
            },
            "qualitative_analysis": {
                "method": "Agentic Literature Search (LangGraph + PubMed)",
                "llm_summary": agent_summary
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))