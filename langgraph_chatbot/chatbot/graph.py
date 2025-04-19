from langgraph.graph import StateGraph
from .state import ChatState, ChatMessage
from .nodes import chatbot_node, excel_analysis_node, excel_formula_node

def build_graph():
    # Create a new graph
    graph = StateGraph(ChatState)
    
    # Add nodes with names that don't conflict with state fields
    graph.add_node("analyzer", excel_analysis_node)  # Changed from "excel_analysis"
    graph.add_node("formula_gen", excel_formula_node)  # Changed from "excel_formula"
    graph.add_node("chatbot", chatbot_node)
    
    # Define the flow - update the edge references
    # First analyze Excel data if available
    graph.add_edge("analyzer", "formula_gen")  # Updated
    # Then generate formulas if needed
    graph.add_edge("formula_gen", "chatbot")  # Updated
    
    # Define the entry point
    graph.set_entry_point("analyzer")  # Updated
    
    # Compile the graph
    return graph.compile()
