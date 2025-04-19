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

def process_excel(file):
    """Process the uploaded Excel file and return a summary."""
    if file is None:
        return "No file uploaded.", None
    
    try:
        # Read the Excel file - check for multiple sheets
        excel_file = pd.ExcelFile(file.name)
        sheet_names = excel_file.sheet_names
        
        if len(sheet_names) > 1:
            # Multiple sheets - read all into a dictionary of DataFrames
            dfs = {sheet_name: pd.read_excel(file.name, sheet_name=sheet_name) 
                   for sheet_name in sheet_names}
            
            # Generate a summary of the Excel file
            summary = f"Excel file uploaded successfully with {len(sheet_names)} sheets!\n\n"
            summary += f"Sheets: {', '.join(sheet_names)}\n\n"
            
            # Add a brief preview of each sheet
            for sheet_name, df in dfs.items():
                num_rows, num_cols = df.shape
                summary += f"Sheet '{sheet_name}':\n"
                summary += f"  - {num_rows} rows × {num_cols} columns\n"
                summary += f"  - Columns: {', '.join(df.columns.tolist())}\n"
                summary += f"  - Preview (first 3 rows):\n"
                summary += df.head(3).to_string() + "\n\n"
            
            return summary, dfs
        else:
            # Single sheet - read as a DataFrame
            df = pd.read_excel(file.name)
            
            # Generate a summary of the Excel file
            num_rows, num_cols = df.shape
            columns = df.columns.tolist()
            
            summary = f"Excel file uploaded successfully!\n\n"
            summary += f"File contains {num_rows} rows × {num_cols} columns.\n"
            summary += f"Columns: {', '.join(columns)}\n\n"
            
            # Add a preview of the data
            summary += "Preview of the data:\n"
            summary += df.head(5).to_string()
            
            return summary, df
    
    except Exception as e:
        return f"Error processing Excel file: {str(e)}", None

def respond(message, history, file=None):
    # Process file if uploaded
    file_info = ""
    df = None
    sheet_names = []
    
    if file is not None:
        file_info, df = process_excel(file)
        # Add file information to the message
        message = f"{message}\n\nI've uploaded an Excel file with the following information:\n{file_info}"
        
        # Extract sheet names if it's a dictionary of DataFrames
        if isinstance(df, dict):
            sheet_names = list(df.keys())
    
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
    
    # Check if the user is identifying sheets
    settlements_sheets = None
    if file is not None and sheet_names:
        # Look for sheet identification in the message
        message_lower = message.lower()
        
        # Initialize dictionaries to store potential matches
        sales_matches = {}
        policy_matches = {}
        
        # Check for sales sheet identification
        sales_keywords = ["sales", "sales sheet", "판매", "판매 시트", "처리일"]
        for keyword in sales_keywords:
            if keyword in message_lower:
                # Look for sheet names mentioned near the keyword
                for sheet in sheet_names:
                    if sheet.lower() in message_lower:
                        # Calculate proximity (simple version: are they in the same sentence?)
                        sentences = message_lower.split('.')
                        for sentence in sentences:
                            if keyword in sentence and sheet.lower() in sentence:
                                sales_matches[sheet] = sales_matches.get(sheet, 0) + 1
        
        # Check for policy sheet identification
        policy_keywords = ["policy", "policy sheet", "정책", "정책 시트", "번들결합분류"]
        for keyword in policy_keywords:
            if keyword in message_lower:
                # Look for sheet names mentioned near the keyword
                for sheet in sheet_names:
                    if sheet.lower() in message_lower:
                        # Calculate proximity
                        sentences = message_lower.split('.')
                        for sentence in sentences:
                            if keyword in sentence and sheet.lower() in sentence:
                                policy_matches[sheet] = policy_matches.get(sheet, 0) + 1
        
        # Determine the most likely sheets
        sales_sheet = max(sales_matches.items(), key=lambda x: x[1])[0] if sales_matches else None
        policy_sheet = max(policy_matches.items(), key=lambda x: x[1])[0] if policy_matches else None
        
        # If sheets were identified, create the SettlementSheets object
        if sales_sheet or policy_sheet:
            from chatbot.state import SettlementSheets
            settlements_sheets = {}
            if sales_sheet:
                settlements_sheets["sales"] = sales_sheet
            if policy_sheet:
                settlements_sheets["policy"] = policy_sheet
    
    # Create state and invoke graph
    current_state = ChatState(
        messages=messages, 
        uploaded_file_data=df,
        settlements_sheets=settlements_sheets
    )
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
    chatbot=gr.Chatbot(height=600, type="messages"),  # Add this parameter
    title="Excel Settlement Calculator Assistant",
    description="Ask me about Excel formulas, settlement calculations, or financial spreadsheets! You can also upload an Excel file for analysis.",
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
