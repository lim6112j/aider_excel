from langchain_openai import ChatOpenAI
from .state import ChatState, ChatMessage

def chatbot_node(state: ChatState) -> ChatState:
    """Process the messages and generate a response."""
    # Initialize the LLM
    llm = ChatOpenAI(temperature=0.7)
    
    # Define system prompt for Excel professional
    system_prompt = """You are an Excel professional who specializes in calculating settlements.
    You can help with:
    - Creating formulas for financial calculations
    - Setting up spreadsheets for settlement tracking
    - Calculating payment distributions
    - Reconciling accounts
    - Analyzing financial data
    - Automating settlement processes with Excel functions
    - Explaining Excel techniques for financial calculations
    
    Provide clear, step-by-step instructions when explaining formulas or calculations.
    When appropriate, show the actual Excel formula syntax that would be used.
    If you need specific numbers or details to provide an accurate calculation, ask for them.
    """
    
    # Format messages for the LLM, including the system prompt
    formatted_messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add user and assistant messages
    for msg in state.messages:
        formatted_messages.append({"role": msg.role, "content": msg.content})
    
    # Generate response
    response = llm.invoke(formatted_messages)
    
    # Update state with the response
    new_message = ChatMessage(role="assistant", content=response.content)
    
    return ChatState(
        messages=state.messages + [new_message],
        current_response=response.content
    )
