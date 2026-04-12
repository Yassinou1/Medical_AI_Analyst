# Medical AI Analyst

A hybrid machine learning and generative AI project for **predicting and analyzing adverse drug events (Pharmacovigilance)**, implemented as a microservice architecture.

## Repository Contents

- `main.py` – Core FastAPI backend handling API requests and routing.
- `app.py` – Streamlit web dashboard for interactive visualization and benchmarking of adverse events.
- `train_lstm.py` – Standalone script for training the multivariate LSTM neural network on medical data.
- `eval_model.py` – Script for quantitative evaluation and backtesting of the model's accuracy.
- `services/` – Directory containing the core logic:
  - `lstm_service.py` – Handles quantitative predictions based on historical trends.
  - `agent_service.py` – Executes autonomous literature research using LangGraph and a local LLM.
  - `openfda_service.py` – Manages the data pipeline connecting to the public OpenFDA API.
- `requirements.txt` – List of necessary Python dependencies.

## Description

This project explores predictive modeling combined with an agentic AI workflow for the healthcare sector.  
The goal is to provide a quantitative prediction of upcoming adverse drug events using a multivariate LSTM model, and enrich it with an evidence-based clinical summary using an autonomous AI agent (LangGraph) equipped with medical literature access.  
It serves as an end-to-end portfolio project applying modern MLOps, deep learning, and GenAI techniques to Pharmacovigilance.

## Data Sources

The data used in this project is sourced live from two primary APIs:

1. **OpenFDA API:** Provides historical, real-time data on reported adverse drug events (used for training the LSTM and feature engineering).  
   -> [OpenFDA Documentation](https://open.fda.gov/apis/drug/event/)

2. **PubMed API:** Serves as the tool for the LangGraph agent to autonomously search for the latest medical studies, case reports, and contraindications.

## Installation & Setup

This project requires **Python 3.11** and a local installation of [Ollama](https://ollama.com/) running the `mistral` model.

**1. Clone and Environment Setup**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
