from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from .state import ChatState, ChatMessage
import pandas as pd
import numpy as np

def excel_analysis_node(state: ChatState) -> ChatState:
    """Analyze Excel data from the uploaded file."""
    # Check if there's an Excel file to analyze
    if state.uploaded_file_data is None:
        # No file to analyze, just return the state unchanged
        return state
    
    # Get the latest user message
    latest_user_message = next((msg.content for msg in reversed(state.messages) 
                               if msg.role == "user"), "")
    
    # Get the DataFrame or dict of DataFrames
    df_data = state.uploaded_file_data
    query = latest_user_message.lower()
    
    # Check if we have multiple sheets (df_data is a dict of DataFrames)
    is_multi_sheet = isinstance(df_data, dict)
    
    try:
        # If it's a single DataFrame (not a dict), convert to dict format for consistent handling
        if not is_multi_sheet:
            df_data = {"Sheet1": df_data}
        
        # Basic file information
        sheet_names = list(df_data.keys())
        
        # If query mentions a specific sheet, focus on that sheet
        mentioned_sheet = None
        for sheet in sheet_names:
            if sheet.lower() in query:
                mentioned_sheet = sheet
                break
        
        # Basic statistics or sheet listing
        if "summary" in query or "statistics" in query or "sheets" in query:
            summary = f"Excel File Summary:\n"
            
            if len(sheet_names) > 1:
                summary += f"The file contains {len(sheet_names)} sheets: {', '.join(sheet_names)}\n\n"
            
            # If a specific sheet is mentioned or there's only one sheet, show its details
            if mentioned_sheet or len(sheet_names) == 1:
                sheet_to_analyze = mentioned_sheet if mentioned_sheet else sheet_names[0]
                df = df_data[sheet_to_analyze]
                
                summary += f"Sheet '{sheet_to_analyze}':\n"
                summary += f"  - Rows: {df.shape[0]}\n"
                summary += f"  - Columns: {df.shape[1]}\n"
                summary += f"  - Column names: {', '.join(df.columns.tolist())}\n\n"
                
                # Numeric columns statistics for the selected sheet
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    summary += f"Numeric Columns Statistics for sheet '{sheet_to_analyze}':\n"
                    stats = df[numeric_cols].describe().to_string()
                    summary += f"{stats}\n\n"
                
                # Preview of the data
                summary += f"Preview of sheet '{sheet_to_analyze}':\n"
                summary += df.head(5).to_string() + "\n\n"
            else:
                # Show basic info for all sheets
                for sheet_name in sheet_names:
                    df = df_data[sheet_name]
                    summary += f"Sheet '{sheet_name}':\n"
                    summary += f"  - Rows: {df.shape[0]}\n"
                    summary += f"  - Columns: {df.shape[1]}\n"
                    summary += f"  - Column names: {', '.join(df.columns.tolist())}\n\n"
            
            analysis_result = summary
        
        # Column analysis - need to specify which sheet if multiple
        elif "column" in query:
            if mentioned_sheet:
                df = df_data[mentioned_sheet]
                sheet_to_analyze = mentioned_sheet
            elif len(sheet_names) == 1:
                df = df_data[sheet_names[0]]
                sheet_to_analyze = sheet_names[0]
            else:
                return ChatState(
                    messages=state.messages,
                    current_response=state.current_response,
                    uploaded_file_data=state.uploaded_file_data,
                    excel_analysis_result="Please specify which sheet you want to analyze. Available sheets: " + 
                                         ", ".join(sheet_names)
                )
            
            # Find which column was mentioned
            mentioned_col = next((col for col in df.columns if col.lower() in query), None)
            
            if mentioned_col:
                col_analysis = f"Analysis of column '{mentioned_col}' in sheet '{sheet_to_analyze}':\n"
                
                # Check if numeric
                if pd.api.types.is_numeric_dtype(df[mentioned_col]):
                    col_analysis += f"Data type: Numeric\n"
                    col_analysis += f"Min: {df[mentioned_col].min()}\n"
                    col_analysis += f"Max: {df[mentioned_col].max()}\n"
                    col_analysis += f"Mean: {df[mentioned_col].mean()}\n"
                    col_analysis += f"Sum: {df[mentioned_col].sum()}\n"
                else:
                    col_analysis += f"Data type: Non-numeric\n"
                    col_analysis += f"Unique values: {df[mentioned_col].nunique()}\n"
                    
                    # Show value counts for categorical data
                    value_counts = df[mentioned_col].value_counts().head(10).to_string()
                    col_analysis += f"Top values:\n{value_counts}\n"
                
                analysis_result = col_analysis
            else:
                analysis_result = f"Column not found in sheet '{sheet_to_analyze}'. Available columns: {', '.join(df.columns)}"
        
        # Calculate sum - need to specify which sheet if multiple
        elif "sum" in query or "total" in query:
            if mentioned_sheet:
                df = df_data[mentioned_sheet]
                sheet_to_analyze = mentioned_sheet
            elif len(sheet_names) == 1:
                df = df_data[sheet_names[0]]
                sheet_to_analyze = sheet_names[0]
            else:
                return ChatState(
                    messages=state.messages,
                    current_response=state.current_response,
                    uploaded_file_data=state.uploaded_file_data,
                    excel_analysis_result="Please specify which sheet you want to analyze. Available sheets: " + 
                                         ", ".join(sheet_names)
                )
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                analysis_result = f"No numeric columns found in sheet '{sheet_to_analyze}' to sum."
            else:
                # Try to identify which column to sum
                mentioned_col = next((col for col in numeric_cols if col.lower() in query), None)
                
                if mentioned_col:
                    total = df[mentioned_col].sum()
                    analysis_result = f"Sum of '{mentioned_col}' in sheet '{sheet_to_analyze}': {total}"
                else:
                    # Sum all numeric columns
                    results = f"Sums of all numeric columns in sheet '{sheet_to_analyze}':\n"
                    for col in numeric_cols:
                        results += f"{col}: {df[col].sum()}\n"
                    analysis_result = results
        
        # Generic query - return basic info about sheets
        else:
            if len(sheet_names) > 1:
                # If no specific analysis is requested but we have multiple sheets, provide sheet info
                sheet_info = "This Excel file contains multiple sheets:\n"
                for sheet_name in sheet_names:
                    df = df_data[sheet_name]
                    sheet_info += f"- '{sheet_name}': {df.shape[0]} rows × {df.shape[1]} columns\n"
                sheet_info += "\nYou can ask for analysis of a specific sheet by mentioning its name."
                analysis_result = sheet_info
            else:
                # If no specific analysis is requested and only one sheet, don't add any analysis
                return state
            
    except Exception as e:
        analysis_result = f"Error analyzing Excel file: {str(e)}"
    
    # Add the analysis result to the state for the chatbot to use
    state.excel_analysis_result = analysis_result
    return state

def excel_formula_node(state: ChatState) -> ChatState:
    """Generate Excel formulas based on the user query."""
    # Get the latest user message
    latest_user_message = next((msg.content for msg in reversed(state.messages) 
                               if msg.role == "user"), "")
    
    query = latest_user_message.lower()
    
    # Only generate formulas if specifically requested
    if not any(keyword in query for keyword in ["formula", "calculate", "computation", "excel function"]):
        return state
    
    if "pro-rata" in query or "prorate" in query:
        formula = """
        Pro-rata Settlement Formula:
        
        =Amount * (Days_Used / Total_Period_Days)
        
        Example in Excel:
        If the total amount is in cell A1, days used in B1, and total period days in C1:
        =A1*(B1/C1)
        
        This calculates the proportional amount based on the time period used.
        """
        
    elif "interest" in query:
        formula = """
        Interest Calculation Formula:
        
        Simple Interest: =Principal * Rate * Time
        
        Example in Excel:
        If principal is in cell A1, annual interest rate in B1 (as decimal), and time in years in C1:
        =A1*B1*C1
        
        For compound interest:
        =Principal * (1 + Rate)^Time
        
        In Excel:
        =A1*(1+B1)^C1
        
        For daily compounding:
        =Principal * (1 + Rate/365)^(365*Time)
        
        In Excel:
        =A1*(1+B1/365)^(365*C1)
        """
        
    elif "payment" in query or "installment" in query:
        formula = """
        Payment Calculation Formula (for loans or installments):
        
        PMT function: =PMT(rate, nper, pv, [fv], [type])
        
        Where:
        - rate: Interest rate per period
        - nper: Total number of payment periods
        - pv: Present value (loan amount)
        - fv: Future value (optional, default is 0)
        - type: When payments are due (0=end of period, 1=beginning of period)
        
        Example in Excel:
        If monthly interest rate is in A1, number of payments in B1, and loan amount in C1:
        =PMT(A1, B1, C1)
        
        Note: For an annual interest rate, divide by 12 for monthly payments:
        =PMT(AnnualRate/12, NumberOfMonths, LoanAmount)
        """
        
    else:
        formula = """
        Common Excel Formulas for Settlements:
        
        1. SUM: =SUM(range) - Adds all numbers in a range
           Example: =SUM(A1:A10)
        
        2. AVERAGE: =AVERAGE(range) - Calculates the average of numbers
           Example: =AVERAGE(B1:B20)
        
        3. IF: =IF(condition, value_if_true, value_if_false) - Conditional logic
           Example: =IF(A1>1000, "High", "Low")
        
        4. VLOOKUP: =VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])
           Example: =VLOOKUP(A1, B1:C10, 2, FALSE)
        
        5. DATE: =DATE(year, month, day) - Creates a date value
           Example: =DATE(2023, 12, 31)
        
        6. DATEDIF: =DATEDIF(start_date, end_date, unit) - Calculates difference between dates
           Example: =DATEDIF(A1, B1, "D") for days
        """
    
    # Add the formula to the state for the chatbot to use
    state.excel_formula = formula
    return state

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
    
    You can analyze Excel files with multiple sheets. When a user uploads a file with multiple sheets,
    you can provide information about specific sheets or columns by name.
    
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
    
    # If there's Excel analysis result, add it to the context
    if hasattr(state, 'excel_analysis_result') and state.excel_analysis_result:
        formatted_messages.append({
            "role": "system", 
            "content": f"Excel file analysis result:\n{state.excel_analysis_result}"
        })
    
    # If there's Excel formula, add it to the context
    if hasattr(state, 'excel_formula') and state.excel_formula:
        formatted_messages.append({
            "role": "system", 
            "content": f"Relevant Excel formula:\n{state.excel_formula}"
        })
    
    # Generate response
    response = llm.invoke(formatted_messages)
    
    # Update state with the response
    new_message = ChatMessage(role="assistant", content=response.content)
    
    return ChatState(
        messages=state.messages + [new_message],
        current_response=response.content,
        uploaded_file_data=state.uploaded_file_data  # Preserve the uploaded file data
    )
