from langchain_openai import ChatOpenAI
from .state import ChatState, ChatMessage

def chatbot_node(state: ChatState) -> ChatState:
    """Process the messages and generate a response."""
    # Initialize the LLM
    llm = ChatOpenAI(temperature=0.7)
    
    # Format messages for the LLM
    formatted_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in state.messages
    ]
    
    # Generate response
    response = llm.invoke(formatted_messages)
    
    # Update state with the response
    new_message = ChatMessage(role="assistant", content=response.content)
    
    return ChatState(
        messages=state.messages + [new_message],
        current_response=response.content
    )
