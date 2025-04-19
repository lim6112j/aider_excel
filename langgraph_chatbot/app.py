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
        
        # Access the result using dictionary keys
        if "current_response" in result:
            print(f"Bot: {result['current_response']}")
        elif "messages" in result:
            # Get the last assistant message
            messages = result["messages"]
            assistant_messages = [msg for msg in messages if msg.role == "assistant"]
            if assistant_messages:
                print(f"Bot: {assistant_messages[-1].content}")
            else:
                print("Bot: No response generated.")
        else:
            print("Bot: Could not retrieve response.")
        
        # Update state for next iteration
        state = ChatState(
            messages=result["messages"] if "messages" in result else [],
            current_response=result.get("current_response")
        )

if __name__ == "__main__":
    main()
