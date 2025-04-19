import os
from chatbot.graph import build_graph
from chatbot.state import ChatState, ChatMessage

def main():
    # Make sure OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Build the graph
    graph = build_graph()
    
    print("Chatbot initialized. Type 'exit' to quit.")
    
    # Initial state
    state = ChatState(messages=[])
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            break
        
        # Add user message to state
        user_message = ChatMessage(role="user", content=user_input)
        current_state = ChatState(messages=state.messages + [user_message])
        
        # Process through the graph
        result = graph.invoke(current_state)
        
        # Debug the result structure
        print(f"Debug - Result type: {type(result)}")
        print(f"Debug - Result keys: {result.keys() if hasattr(result, 'keys') else 'No keys method'}")
        
        # Try to extract the state based on the actual structure
        # For now, let's assume the result itself is the state
        state = result
        
        # Display response
        if hasattr(state, 'current_response'):
            print(f"Bot: {state.current_response}")
        else:
            print("Bot response not found in expected format.")
            # Try to find messages in the result
            if hasattr(state, 'messages') and state.messages:
                # Get the last assistant message
                assistant_messages = [msg for msg in state.messages if msg.role == "assistant"]
                if assistant_messages:
                    print(f"Bot: {assistant_messages[-1].content}")

if __name__ == "__main__":
    main()
