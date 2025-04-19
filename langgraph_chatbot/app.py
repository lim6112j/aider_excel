import os
import gradio as gr
import pandas as pd
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
# Global variable to store the uploaded file data
uploaded_file_data = None

def process_excel(file):
    """Process the uploaded Excel file and return a summary."""
    global uploaded_file_data
    
    if file is None:
        return "No file uploaded."
    
    try:
        # Read the Excel file
        df = pd.read_excel(file.name)
        
        # Store the dataframe for later use
        uploaded_file_data = df
        
        # Generate a summary of the Excel file
        num_rows, num_cols = df.shape
        columns = df.columns.tolist()
        
        summary = f"Excel file uploaded successfully!\n\n"
        summary += f"File contains {num_rows} rows and {num_cols} columns.\n"
        summary += f"Columns: {', '.join(columns)}\n\n"
        
        # Add a preview of the data
        summary += "Preview of the data:\n"
        summary += df.head(5).to_string()
        
        return summary
    
    except Exception as e:
        return f"Error processing Excel file: {str(e)}"

def respond(message, history, file=None):
    global uploaded_file_data
    
    # Process file if uploaded
    file_info = ""
    if file is not None:
        file_info = process_excel(file)
        # Add file information to the message
        message = f"{message}\n\nI've uploaded an Excel file with the following information:\n{file_info}"
    
    # Convert Gradio history to our ChatState format
    messages = []
    for msg in history:
        # Handle both possible formats of history
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            # Already in the right format
            role = msg["role"]
            content = msg["content"]
            messages.append(ChatMessage(role=role, content=content))
        elif isinstance(msg, list) and len(msg) == 2:
            # Old tuple format [user_msg, assistant_msg]
            if msg[0]:  # User message
                messages.append(ChatMessage(role="user", content=msg[0]))
            if msg[1]:  # Assistant message
                messages.append(ChatMessage(role="assistant", content=msg[1]))
            continue
    
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
        result_messages = result["messages"]
        assistant_messages = [msg for msg in result_messages if msg.role == "assistant"]
        if assistant_messages:
            bot_response = assistant_messages[-1].content
        else:
            bot_response = "No response generated."
    else:
        bot_response = "Could not retrieve response."
    
    # Return in the format expected by Gradio chatbot with type='messages'
    return bot_response

# Create Gradio interface with ChatInterface
demo = gr.ChatInterface(
    fn=respond,
    title="Excel Settlement Calculator Assistant",
    description="Ask me about Excel formulas, settlement calculations, or financial spreadsheets!",
    examples=[
        # Format: [message, file]
        ["How do I calculate a pro-rata settlement distribution in Excel?", None],
        ["What formula should I use to calculate interest on outstanding settlements?", None],
        ["How can I set up a spreadsheet to track multiple settlement payments?", None],
        ["What's the best way to reconcile settlement accounts in Excel?", None]
    ],
    additional_inputs=[
        gr.File(
            file_types=[".xlsx", ".xls", ".csv"],
            label="Upload Excel File"
        )
    ],
    theme="soft"
)

# Launch the app
if __name__ == "__main__":
    # Add pandas to requirements.txt
    demo.launch(share=True)  # Set share=False in production
