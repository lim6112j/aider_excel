from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
import pandas as pd
import numpy as np

class ExcelAnalysisTool(BaseTool):
    name = "excel_analysis"
    description = """
    Analyze Excel data to extract insights, perform calculations, or transform data.
    Input should be a specific analysis request related to the uploaded Excel file.
    """
    
    def _run(self, query: str, **kwargs) -> str:
        """Execute the Excel analysis based on the query."""
        # Get uploaded_file_data from the state passed in kwargs
        uploaded_file_data = kwargs.get("uploaded_file_data")
        
        if uploaded_file_data is None:
            return "No Excel file has been uploaded. Please upload a file first."
        
        df = uploaded_file_data
        
        try:
            # Basic statistics
            if "summary" in query.lower() or "statistics" in query.lower():
                summary = f"Excel File Summary:\n"
                summary += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
                
                # Numeric columns statistics
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    summary += "Numeric Columns Statistics:\n"
                    stats = df[numeric_cols].describe().to_string()
                    summary += f"{stats}\n\n"
                
                return summary
            
            # Column analysis
            elif "column" in query.lower() and any(col.lower() in query.lower() for col in df.columns):
                # Find which column was mentioned
                mentioned_col = next((col for col in df.columns if col.lower() in query.lower()), None)
                
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
                    
                    return col_analysis
                else:
                    return f"Column not found in the Excel file. Available columns: {', '.join(df.columns)}"
            
            # Calculate sum
            elif "sum" in query.lower() or "total" in query.lower():
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_cols:
                    return "No numeric columns found in the Excel file to sum."
                
                # Try to identify which column to sum
                mentioned_col = next((col for col in numeric_cols if col.lower() in query.lower()), None)
                
                if mentioned_col:
                    total = df[mentioned_col].sum()
                    return f"Sum of '{mentioned_col}': {total}"
                else:
                    # Sum all numeric columns
                    results = "Sums of all numeric columns:\n"
                    for col in numeric_cols:
                        results += f"{col}: {df[col].sum()}\n"
                    return results
            
            # Generic query - return basic info
            else:
                return f"Excel file loaded with {df.shape[0]} rows and {df.shape[1]} columns. Please specify what analysis you'd like to perform."
                
        except Exception as e:
            return f"Error analyzing Excel file: {str(e)}"

    def _arun(self, query: str):
        """Async implementation would go here."""
        raise NotImplementedError("This tool does not support async")


class ExcelFormulaTool(BaseTool):
    name = "excel_formula_generator"
    description = """
    Generate Excel formulas for specific calculations or data manipulations.
    Input should be a description of the calculation or formula needed.
    """
    
    def _run(self, query: str) -> str:
        """Generate Excel formulas based on the query."""
        # This is a simplified version - in a real implementation, you might use the LLM
        # to generate formulas based on the specific query and data
        
        if "pro-rata" in query.lower() or "prorate" in query.lower():
            return """
            Pro-rata Settlement Formula:
            
            =Amount * (Days_Used / Total_Period_Days)
            
            Example in Excel:
            If the total amount is in cell A1, days used in B1, and total period days in C1:
            =A1*(B1/C1)
            
            This calculates the proportional amount based on the time period used.
            """
            
        elif "interest" in query.lower():
            return """
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
            
        elif "payment" in query.lower() or "installment" in query.lower():
            return """
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
            return """
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
            
            Please specify what type of calculation you need for more specific formulas.
            """
    
    def _arun(self, query: str):
        """Async implementation would go here."""
        raise NotImplementedError("This tool does not support async")
