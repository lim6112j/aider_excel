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
    
    # Get the DataFrame
    df = state.uploaded_file_data
    query = latest_user_message.lower()
    
    try:
        # Basic statistics
        if "summary" in query or "statistics" in query:
            summary = f"Excel File Summary:\n"
            summary += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            
            # Numeric columns statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                summary += "Numeric Columns Statistics:\n"
                stats = df[numeric_cols].describe().to_string()
                summary += f"{stats}\n\n"
            
            analysis_result = summary
        
        # Column analysis
        elif "column" in query and any(col.lower() in query for col in df.columns):
            # Find which column was mentioned
            mentioned_col = next((col for col in df.columns if col.lower() in query), None)
            
            if mentioned_col:
                col_analysis = f"Analysis of column '{mentioned_col}':\n"
                
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
                analysis_result = f"Column not found in the Excel file. Available columns: {', '.join(df.columns)}"
        
        # Calculate sum
        elif "sum" in query or "total" in query:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                analysis_result = "No numeric columns found in the Excel file to sum."
            else:
                # Try to identify which column to sum
                mentioned_col = next((col for col in numeric_cols if col.lower() in query), None)
                
                if mentioned_col:
                    total = df[mentioned_col].sum()
                    analysis_result = f"Sum of '{mentioned_col}': {total}"
                else:
                    # Sum all numeric columns
                    results = "Sums of all numeric columns:\n"
                    for col in numeric_cols:
                        results += f"{col}: {df[col].sum()}\n"
                    analysis_result = results
        
        # Generic query - return basic info
        else:
            # If no specific analysis is requested, don't add any analysis
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
