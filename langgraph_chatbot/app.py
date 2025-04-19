import os
import gradio as gr
from chatbot.graph import build_graph
from chatbot.state import ChatState, ChatMessage

# Make sure OpenAI API key is set
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    # Note: In production, you should handle this more securely

# Build the graph
graph = build_graph()

# Initialize chat history for Gradio
chat_history = []

def respond(message, history):
    # Convert Gradio history to our ChatState format
    messages = []
    for human_msg, ai_msg in history:
        if human_msg:
            messages.append(ChatMessage(role="user", content=human_msg))
        if ai_msg:
            messages.append(ChatMessage(role="assistant", content=ai_msg))
    
    # Add the current message
    messages.append(ChatMessage(role="user", content=message))
    
    # Create state and invoke graph
    current_state = ChatState(messages=messages)
    result = graph.invoke(current_state)
    
    # Extract response
    if "current_response" in result:
        bot_response = result["current_response"]
    elif "messages" in result:
        # Get the last assistant message
        messages = result["messages"]
        assistant_messages = [msg for msg in messages if msg.role == "assistant"]
        if assistant_messages:
            bot_response = assistant_messages[-1].content
        else:
            bot_response = "No response generated."
    else:
        bot_response = "Could not retrieve response."
    
    return bot_response

# Create Gradio interface
demo = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=600),
    title="LangGraph Chatbot",
    description="Ask me anything!",
    theme="soft",
    examples=["Hello, how are you?", "What can you help me with?", "Tell me a joke."],
    cache_examples=False,
)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)  # Set share=False in production
