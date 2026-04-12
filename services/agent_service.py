from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

@tool
def search_medical_literature(query: str) -> str:
    """Sucht in PubMed nach medizinischen Studien, Fallberichten und Kontraindikationen."""
    search = PubmedQueryRun()
    return search.invoke(query)

tools = [search_medical_literature]
tool_node = ToolNode(tools)

llm = ChatOllama(model="mistral", temperature=0)
llm_with_tools = llm.bind_tools(tools)

class AgentService:
    def __init__(self):
        self.workflow = self._build_graph()

    def _build_graph(self):
        def agent_node(state: AgentState):
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "continue"
            return "end"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()

    def analyze_with_agent(self, drug_name: str, predicted_events: int) -> str:
        system_prompt = f"""
        Du bist ein klinischer KI-Analyst für Pharmakovigilanz und Toxikologie.
        Wirkstoff: {drug_name}
        Unser LSTM-Algorithmus schlägt Alarm: Er prognostiziert für nächste Woche {predicted_events} gemeldete Nebenwirkungen.
        
        Deine Aufgabe: Erkläre diesen potenziellen Anstieg. Nutze dein PubMed-Tool, um nach speziellen, selteneren oder neuen Risikofaktoren zu suchen.
        
        STRIKTE OUTPUT-REGELN:
        1. KEIN BASISWISSEN: Erwähne NIEMALS bekannte Standard-Nebenwirkungen (wie Magen-Darm-Blutungen, Übelkeit oder leichte Allergien). Wir wissen das bereits.
        2. FOKUS AUF ANOMALIEN: Nenne stattdessen spezifische Medikamenten-Interaktionen (Wechselwirkungen mit anderen Wirkstoffen), Risiken für sehr spezifische Patientengruppen (z.B. Niereninsuffizienz) oder saisonale/neue toxikologische Erkenntnisse.
        3. FORMAT: Antworte AUSSCHLIESSLICH in 2 bis 3 kurzen Bulletpoints auf Deutsch.
        4. ZITATION: Autor und Jahr in Klammern in den Satz integrieren (z.B. "(Müller et al., 2023)").
        """
        
        inputs = {"messages": [HumanMessage(content=system_prompt)]}
        result = self.workflow.invoke(inputs)
        return result["messages"][-1].content