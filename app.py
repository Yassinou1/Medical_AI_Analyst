"""Streamlit dashboard for the Medical AI Analyst."""

import logging
import streamlit as st
import requests
import pandas as pd
import time

from config import settings

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Medical AI Analyst", layout="wide")

# --- Sidebar / Health Check ---
with st.sidebar:
    st.title("Medical AI Analyst")
    
    st.divider()
    st.subheader("Systemstatus")
    try:
        health_resp = requests.get(f"http://{settings.API_HOST}:{settings.API_PORT}/health", timeout=2)
        if health_resp.status_code == 200:
            st.success("Backend: Online")
        else:
            st.error("Backend: Fehler")
    except Exception:
        st.error("Backend: Offline")
    
    st.divider()
    st.markdown("**Methoden:**\n- Statistische Anomalieerkennung\n- LangGraph Agent (Mistral)\n- Structured JSON Outputs\n- OpenFDA & PubMed API")

# --- Main App ---
st.title("Medikamenten-Risikoüberwachung (Pharmacovigilance)")
st.markdown("Dieses Dashboard nutzt **Statistische Anomalieerkennung (Z-Scores)** zur Identifikation von Meldespitzen und **Evidenzbasierte Literaturrecherche** (Agentic AI) zur Ursachenfindung.")

st.divider()

drug_name = st.text_input("Wirkstoff eingeben:", value="ibuprofen", max_chars=30).lower()

if st.button("Klinische Analyse starten", type="primary"):
    with st.spinner(f"Analysiere FDA-Daten und PubMed-Literatur für {drug_name}..."):
        try:
            response = requests.post(
                f"http://{settings.API_HOST}:{settings.API_PORT}/analyze",
                json={"drug_name": drug_name},
                timeout=120  # Agent needs time to research
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # --- Row 1: Anomalies ---
                st.subheader("Quantitative Überwachung (Letzte Woche)")
                col1, col2, col3 = st.columns(3)
                
                quant = data['quantitative_analysis']
                is_anomaly = quant.get('is_anomaly', False)
                
                col1.metric("Status", quant.get('status', 'Unbekannt'))
                col2.metric("Neu Gemeldete Fälle", quant.get('current_count', 0), delta=f"Z-Score: {quant.get('z_score', 0)}", delta_color="inverse" if is_anomaly else "normal")
                col3.metric("Historischer Durchschnitt", quant.get('baseline_mean', 0))
                
                if is_anomaly:
                    st.error("Signifikante statistische Abweichung im Meldevolumen erkannt!")
                else:
                    st.success("Meldevolumen im normalen statistischen Bereich.")

                st.divider()
                
                # --- Row 2: AI Agent ---
                st.subheader("Klinische KI-Analyse (PubMed)")
                qual = data['qualitative_analysis']
                
                st.info(qual.get('summary', 'Keine Zusammenfassung verfügbar.'))
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Risikofaktoren / Interaktionen:**")
                    for rf in qual.get('risk_factors', []):
                        st.markdown(f"- {rf}")
                with c2:
                    st.markdown("**Bekannte Kontraindikationen:**")
                    for c in qual.get('contraindications', []):
                        st.markdown(f"- {c}")
                        
                raw_conf = float(qual.get('confidence_score', 0.0))
                conf = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
                conf = max(0.0, min(1.0, conf))
                st.progress(conf, text=f"Konfidenz der Erkenntnisse: {conf*100:.0f}%")

                st.divider()
                
                # --- Row 3: Alerts ---
                alerts = data.get('german_regulatory_alerts', [])
                with st.expander(f"Rote-Hand-Briefe / BfArM Warnungen ({len(alerts)} gefunden)", expanded=len(alerts) > 0):
                    if alerts:
                        for a in alerts:
                            st.markdown(f"**[{a['date']}] [{a['title']}]({a['link']})**")
                            st.caption(a['summary'])
                            st.markdown("---")
                    else:
                        st.write("Keine aktuellen Warnungen der deutschen Behörden gefunden.")
                        
            else:
                err_detail = response.json().get('detail', 'Unbekannter Fehler')
                st.error(f"Fehler vom Server ({response.status_code}): {err_detail}")
                
        except requests.exceptions.ConnectionError:
            st.error("Konnte keine Verbindung zum Backend herstellen. Läuft gunicorn/uvicorn?")
        except requests.exceptions.Timeout:
            st.error("Timeout: Die Agenten-Recherche dauerte zu lange.")

st.divider()

# --- Trend Comparison (Microservice compliant) ---
st.subheader("Nebenwirkungs-Trend (Letzte 4 Wochen)")

@st.cache_data(ttl=3600)
def fetch_trend_data(med: str):
    try:
        response = requests.get(
            f"http://{settings.API_HOST}:{settings.API_PORT}/trend",
            params={"drug_name": med},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get('cases_4_weeks', 0)
    except Exception as e:
        logger.error(f"Failed to fetch trend for {med}: {e}")
    return 0

with st.spinner("Lade Vergleichsdaten (Microservice /trend endpoint)..."):
    common_drugs = ["ibuprofen", "acetaminophen", "amoxicillin", "omeprazole"]
    compare_list = [drug_name] + [d for d in common_drugs if d != drug_name]
    
    trend_data = {}
    
    for med in compare_list:
        val = fetch_trend_data(med)
        if val > 0:
            trend_data[med] = val
        time.sleep(0.1) # Be gentle on our backend
    
    if trend_data:
        df_trend = pd.DataFrame(list(trend_data.items()), columns=["Wirkstoff", "Gemeldete Fälle (4 Wochen)"])
        df_trend = df_trend.set_index("Wirkstoff")
        st.bar_chart(df_trend)
    else:
        st.warning("Vergleichsdaten konnten nicht geladen werden.")