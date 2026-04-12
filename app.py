import streamlit as st
import requests
import pandas as pd
from services.openfda_service import MedicalDataService
import time

st.set_page_config(page_title="Medical AI Analyst", layout="centered")

st.title("Medical AI Analyst")
st.markdown("Dieses Dashboard kombiniert **Zeitreihenvorhersagen** (Multivariates LSTM) für gemeldete Nebenwirkungen mit **evidenzbasierter Literaturrecherche** (Agentic AI via PubMed).")

st.divider()

drug_name = st.text_input("Wirkstoff eingeben:", value="ibuprofen", max_chars=30).lower()

if st.button("Klinische Analyse starten", type="primary"):
    with st.spinner(f"Analysiere FDA-Daten und PubMed-Literatur für {drug_name}..."):
        try:
            response = requests.post(
                "http://localhost:8000/analyze",
                json={"drug_name": drug_name}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Quantitative Prognose")
                    st.metric(
                        label="Erwartete Nebenwirkungen (Nächste Woche)",
                        value=data['quantitative_analysis']['predicted_adverse_events']
                    )
                    st.caption(f"Modell: {data['quantitative_analysis']['model']}")
                    
                with col2:
                    st.subheader("Evidenzbasierte Zusammenfassung")
                    st.info(data["qualitative_analysis"]["llm_summary"])
                    method = data["qualitative_analysis"].get("method", "KI Analyse")
                    st.caption(f"**Suchmethode:** {method}")
                        
            else:
                st.error(f"Fehler vom Server: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("Konnte keine Verbindung zum Backend herstellen.")
            st.stop()

    st.divider()

    st.subheader("Nebenwirkungs-Trend (Letzte 4 Wochen)")
    with st.spinner("Lade Vergleichsdaten..."):
        common_drugs = ["ibuprofen", "paracetamol", "amoxicillin", "omeprazole"]
        compare_list = [drug_name] + [d for d in common_drugs if d != drug_name]
        
        trend_data = {}
        service = MedicalDataService()
        
        for med in compare_list:
            try:
                hist = service.fetch_adverse_events_timeseries(med)
                if not hist.empty and len(hist) >= 4:
                    recent_4_weeks = hist['Adverse_Events'].tail(4).sum()
                    trend_data[med] = recent_4_weeks

                # Spam schutz für die API
                time.sleep(1.0) 
                
            except Exception as e:
                print(f"Fehler bei {med}: {e}")
        
        if trend_data:
            df_trend = pd.DataFrame(list(trend_data.items()), columns=["Wirkstoff", "Gemeldete Fälle (4 Wochen)"])
            df_trend = df_trend.set_index("Wirkstoff")
            st.bar_chart(df_trend)
        else:
            st.warning("Vergleichsdaten konnten nicht geladen werden.")