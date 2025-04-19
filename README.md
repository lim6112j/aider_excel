# Excel Settlement Calculator Assistant

A powerful AI chatbot built with LangGraph that specializes in Excel settlement calculations and analysis. This application allows users to upload Excel files (including multi-sheet workbooks) and get expert assistance with formulas, calculations, and data analysis.

## Features

- **Excel Formula Generation**: Get expert guidance on creating formulas for financial calculations, settlement tracking, and more
- **Excel File Analysis**: Upload Excel files and get instant analysis of the data
- **Multi-Sheet Support**: Analyze Excel workbooks with multiple sheets
- **Interactive Web Interface**: User-friendly chat interface built with Gradio
- **Settlement Expertise**: Specialized knowledge in settlement calculations, pro-rata distributions, interest calculations, and more

## Architecture

The application is built using:

- **LangGraph**: For creating a structured, multi-node processing pipeline
- **LangChain**: For building the AI components and interactions
- **OpenAI**: For the underlying language model capabilities
- **Gradio**: For the web interface
- **Pandas**: For Excel file processing and data analysis

The application follows a graph-based architecture with three main nodes:
1. **Excel Analyzer**: Processes uploaded Excel files and extracts insights
2. **Formula Generator**: Creates relevant Excel formulas based on user queries
3. **Chatbot**: Combines analysis, formulas, and user queries to generate helpful responses

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/excel-settlement-assistant.git
cd excel-settlement-assistant
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here  # On Windows: set OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860)

3. Start chatting with the assistant:
   - Ask questions about Excel formulas for settlements
   - Upload Excel files for analysis
   - Get step-by-step guidance on financial calculations

## Example Queries

- "How do I calculate a pro-rata settlement distribution in Excel?"
- "What formula should I use to calculate interest on outstanding settlements?"
- "I've uploaded my settlement data. Can you analyze the 'Payments' column?"
- "How can I set up a spreadsheet to track multiple settlement payments?"
- "What's the best way to reconcile settlement accounts in Excel?"

## Working with Multi-Sheet Excel Files

When you upload an Excel file with multiple sheets:

1. The assistant will provide a summary of all sheets in the workbook
2. You can ask about specific sheets by mentioning their name
3. You can request analysis of specific columns within a sheet
4. You can ask for calculations based on the data in specific sheets

Example: "Can you analyze the 'Amount' column in the 'Q2 Settlements' sheet?"

## Project Structure

```
langgraph_chatbot/
├── app.py                  # Main application file with Gradio interface
├── requirements.txt        # Project dependencies
└── chatbot/
    ├── __init__.py
    ├── graph.py            # LangGraph structure definition
    ├── nodes.py            # Node implementations (analysis, formulas, chat)
    └── state.py            # State definitions
```

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in requirements.txt

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
