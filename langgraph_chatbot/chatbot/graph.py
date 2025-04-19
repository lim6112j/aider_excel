from langgraph.graph import StateGraph
from .state import ChatState, ChatMessage
from .nodes import chatbot_node, excel_analysis_node, excel_formula_node

def build_graph():
    # Create a new graph
    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("excel_analysis", excel_analysis_node)
    graph.add_node("excel_formula", excel_formula_node)
    graph.add_node("chatbot", chatbot_node)
    
    # Define the flow
    # First analyze Excel data if available
    graph.add_edge("excel_analysis", "excel_formula")
    # Then generate formulas if needed
    graph.add_edge("excel_formula", "chatbot")
    
    # Define the entry point
    graph.set_entry_point("excel_analysis")
    
    # Compile the graph
    return graph.compile()
