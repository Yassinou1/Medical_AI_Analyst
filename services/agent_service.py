"""Autonomous agent service for clinical literature research.

Uses LangChain and LangGraph to perform targeted PubMed searches
to explain potential anomalies in adverse drug events. Now upgraded
to return structured JSON (Pydantic models) for API consumption,
and completely asynchronous.
"""

import logging
from typing import TypedDict, Annotated, Sequence, List
import operator
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from config import settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State object for the LangGraph workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]


class ClinicalAnalysis(BaseModel):
    """Structured output expected from the LLM."""
    risk_factors: List[str] = Field(description="Spezifische Risikofaktoren für Patienten oder Medikamentenwechselwirkungen aus der Literatur. WICHTIG: Antworte auf Deutsch.")
    contraindications: List[str] = Field(description="Bekannte oder neu entdeckte Kontraindikationen. WICHTIG: Antworte auf Deutsch.")
    summary: str = Field(description="Kurze Zusammenfassung (2-3 Sätze) der aktuellen Literatur, die die Anomalie erklärt. WICHTIG: Antworte auf Deutsch.")
    confidence_score: float = Field(description="Konfidenz der Ergebnisse von 0.0 bis 1.0 basierend auf der Qualität der Evidenz.")


@tool
def search_medical_literature(query: str) -> str:
    """Search PubMed for medical studies, case reports, and contraindications."""
    search = PubmedQueryRun()
    return search.invoke(query)


class AgentService:
    """Manages the autonomous agent workflow for medical literature analysis."""

    def __init__(self) -> None:
        self.tools = [search_medical_literature]
        self.tool_node = ToolNode(self.tools)
        
        # Standard ChatOllama for reasoning/tool loop
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            temperature=settings.OLLAMA_TEMPERATURE
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Separate LLM call mapped to the Pydantic schema for final formatting
        self.llm_structured = self.llm.with_structured_output(ClinicalAnalysis)
        
        self.workflow = self._build_graph()

    def _build_graph(self):
        async def agent_node(state: AgentState):
            response = await self.llm_with_tools.ainvoke(state["messages"])
            return {"messages": [response]}

        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "continue"
            return "end"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", self.tool_node)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()

    async def analyze_with_agent(self, drug_name: str, current_count: int, z_score: float) -> dict:
        """Executes the agent workflow and formats the output as structured JSON asynchronously.
        
        Args:
            drug_name: The active ingredient.
            current_count: Number of adverse events reported this week.
            z_score: The statistical severity of the anomaly.
            
        Returns:
            Dictionary matching the ClinicalAnalysis schema.
        """
        system_prompt = f"""
        Du bist ein klinischer KI-Analyst für Pharmakovigilanz und Toxikologie.
        Wirkstoff: {drug_name}
        Unser Anomalie-Erkennungssystem meldet einen signifikanten Anstieg von Nebenwirkungsmeldungen.
        Aktuelle wöchentliche Fälle: {current_count} (Z-Score: {z_score}).
        
        Deine Aufgabe: Erkläre diesen potenziellen Anstieg. Nutze dein PubMed-Tool, um nach speziellen, selteneren oder neuen Risikofaktoren zu suchen.
        
        STRIKTE REGELN:
        1. FOKUS AUF ANOMALIEN: Nenne spezifische Medikamenten-Interaktionen oder Risiken für spezifische Patientengruppen.
        2. ZITATION: Autor und Jahr in Klammern in den Text integrieren.
        """
        
        inputs = {"messages": [HumanMessage(content=system_prompt)]}
        
        try:
            # 1. Run the reasoning and search loop
            result = await self.workflow.ainvoke(inputs)
            final_message = result["messages"][-1].content
            
            # 2. Force the output into the structured JSON schema
            logger.info("Formatting agent thoughts into structured ClinicalAnalysis JSON.")
            formatting_prompt = f"Wandle die folgenden Forschungsergebnisse in das vorgegebene JSON-Schema um. WICHTIG: Übersetze alle Erkenntnisse ins Deutsche:\n\n{final_message}"
            structured_result = await self.llm_structured.ainvoke([HumanMessage(content=formatting_prompt)])
            
            return structured_result.model_dump()
            
        except Exception as e:
            logger.exception("Agent service execution failed")
            return {
                "risk_factors": ["Fehler bei der Literaturrecherche"],
                "contraindications": ["Keine Daten abrufbar"],
                "summary": f"Der KI-Agent konnte nicht ausgeführt werden. Stellen Sie sicher, dass Ollama läuft. Details: {str(e)}",
                "confidence_score": 0.0
            }

__all__ = ["AgentService", "ClinicalAnalysis"]