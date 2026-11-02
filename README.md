# Medical AI Analyst

A hybrid machine learning and generative AI project for **predicting and analyzing adverse drug events (Pharmacovigilance)**, implemented as a microservice architecture.

## Repository Contents

- `main.py` – Core FastAPI backend handling API requests, validation, and routing.
- `app.py` – Streamlit web dashboard for interactive visualization and benchmarking of adverse events.
- `train_lstm.py` – Script for training the multivariate LSTM neural network on medical data, with chronological train/test splitting and Early Stopping.
- `eval_model.py` – Script for quantitative evaluation and backtesting of the model's accuracy on the holdout test set.
- `config.py` - Centralized application configuration.
- `services/` – Directory containing the core business logic:
  - `lstm_service.py` – Handles quantitative predictions based on historical trends.
  - `agent_service.py` – Executes autonomous literature research using LangGraph and a local LLM.
  - `openfda_service.py` – Manages the data pipeline connecting to the public OpenFDA API.
  - `german_alerts_service.py` - Fetches regulatory drug safety alerts (Rote-Hand-Briefe) from the BfArM RSS feed.
  - `preprocessing.py` - Shared feature engineering pipeline.

## Description

This project explores predictive modeling combined with an agentic AI workflow for the healthcare sector.  
The goal is to provide a quantitative prediction of upcoming adverse drug events using a multivariate LSTM model, and enrich it with an evidence-based clinical summary using an autonomous AI agent (LangGraph) equipped with medical literature access.  
It serves as an end-to-end portfolio project applying modern MLOps, deep learning, and GenAI techniques to Pharmacovigilance.

## Data Sources

The data used in this project is sourced live from three primary APIs:

1. **OpenFDA API:** Provides historical, real-time data on reported adverse drug events (used for training the LSTM and feature engineering).  
   -> [OpenFDA Documentation](https://open.fda.gov/apis/drug/event/)
2. **PubMed API:** Serves as the tool for the LangGraph agent to autonomously search for the latest medical studies, case reports, and contraindications.
3. **BfArM RSS:** Provides official regulatory safety communications (Rote-Hand-Briefe) from German authorities.

## Installation & Setup

This project requires **Python 3.12** and a local installation of [Ollama](https://ollama.com/) running the `mistral` model.

**1. Clone and Environment Setup**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure Environment**
Copy the example environment file and customize if needed:
```bash
cp .env.example .env
```

**3. Install & Start Ollama (with Mistral)**
Ensure Ollama is installed and the `mistral` model is pulled:
```bash
ollama serve &
ollama pull mistral
```

**4. Train the Model (Optional)**
The repository comes with pre-configured scripts to retrain the model:
```bash
python train_lstm.py --drug ibuprofen
python eval_model.py --drug ibuprofen
```

## Running the Application

You will need three terminal windows to run the full stack:

**Terminal 1: Ollama Server**
```bash
ollama serve
```

**Terminal 2: FastAPI Backend**
```bash
source .venv/bin/activate
uvicorn main:app --reload
```

**Terminal 3: Streamlit Dashboard**
```bash
source .venv/bin/activate
streamlit run app.py
```

Navigate to `http://localhost:8501` to access the dashboard.
