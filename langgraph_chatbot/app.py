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
        
        # Update state - extract the actual state from the result
        state = result["state"]  # Access the state from the result dictionary
        
        # Display response
        print(f"Bot: {state.current_response}")

if __name__ == "__main__":
    main()
