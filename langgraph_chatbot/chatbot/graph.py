from langgraph.graph import StateGraph
from .state import ChatState, ChatMessage
from .nodes import chatbot_node

def build_graph():
    # Create a new graph
    graph = StateGraph(ChatState)
    
    # Add the chatbot node
    graph.add_node("chatbot", chatbot_node)
    
    # Define the entry point
    graph.set_entry_point("chatbot")
    
    # No conditional routing for this simple example
    # The graph will just process through the chatbot node and return
    
    # Compile the graph
    return graph.compile()
