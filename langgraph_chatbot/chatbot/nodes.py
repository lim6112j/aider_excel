from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from .state import ChatState, ChatMessage
from .tools import ExcelAnalysisTool, ExcelFormulaTool

# Initialize tools
excel_analysis_tool = ExcelAnalysisTool()
excel_formula_tool = ExcelFormulaTool()
tools = [excel_analysis_tool, excel_formula_tool]

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
    
    You have access to tools that can analyze Excel files and generate formulas.
    Use these tools when a user uploads an Excel file or asks for specific calculations.
    """
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Format messages for chat history
    chat_history = []
    for msg in state.messages:
        if msg.role == "user":
            chat_history.append(("human", msg.content))
        elif msg.role == "assistant":
            chat_history.append(("ai", msg.content))
    
    # Get the latest user message
    latest_user_message = next((msg.content for msg in reversed(state.messages) 
                               if msg.role == "user"), "")
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Run the agent
    response = agent_executor.invoke({
        "input": latest_user_message,
        "chat_history": chat_history,
        "agent_scratchpad": []
    })
    
    # Extract the response
    bot_response = response["output"]
    
    # Update state with the response
    new_message = ChatMessage(role="assistant", content=bot_response)
    
    return ChatState(
        messages=state.messages + [new_message],
        current_response=bot_response
    )
