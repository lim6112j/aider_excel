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
        if msg["role"] == "user":
            messages.append(ChatMessage(role="user", content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(ChatMessage(role="assistant", content=msg["content"]))
    
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
    
    # Return in the format expected by Gradio chatbot with type='messages'
    return {"role": "assistant", "content": bot_response}

# Create Gradio interface with additional components
with gr.Blocks() as demo:
    gr.Markdown("# Excel Settlement Calculator Assistant")
    gr.Markdown("Ask me about Excel formulas, settlement calculations, or financial spreadsheets!")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=500,
                type='messages'  # Add this parameter to fix the warning
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    container=False,
                    scale=8
                )
                file_upload = gr.File(
                    file_types=[".xlsx", ".xls", ".csv"],
                    label="Upload Excel File",
                    scale=2
                )
            
            with gr.Row():
                submit = gr.Button("Send")
                clear = gr.Button("Clear")
    
    with gr.Accordion("Example Questions", open=False):
        examples = gr.Examples(
            examples=[
                "How do I calculate a pro-rata settlement distribution in Excel?", 
                "What formula should I use to calculate interest on outstanding settlements?",
                "How can I set up a spreadsheet to track multiple settlement payments?",
                "What's the best way to reconcile settlement accounts in Excel?"
            ],
            inputs=msg
        )
    
    # Set up event handlers
    submit_event = submit.click(
        fn=respond,
        inputs=[msg, chatbot, file_upload],
        outputs=chatbot
    ).then(
        fn=lambda: (None, None),  # Clear message and file upload after sending
        outputs=[msg, file_upload]
    )
    
    # Also trigger on Enter key
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, file_upload],
        outputs=chatbot
    ).then(
        fn=lambda: (None, None),
        outputs=[msg, file_upload]
    )
    
    # Clear button functionality
    clear.click(lambda: None, None, chatbot)

# Launch the app
if __name__ == "__main__":
    # Add pandas to requirements.txt
    demo.launch(share=True)  # Set share=False in production
