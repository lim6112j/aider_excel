import pandas as pd
import gradio as gr
from langgraph.graph import StateGraph, END
from typing import Dict, List, TypedDict
import plotly.express as px
from datetime import datetime
import json
import os
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client for Grok 3 API
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise ValueError(
        "XAI_API_KEY environment variable not set. Please set it with your xAI API key.")
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1"
)

# LLM invoke function using Grok 3 API


def llm_invoke(prompt: str, conversation_history: List[Dict] = None) -> Dict:
    try:
        history_messages = [
            {"role": "user", "content": h["query"]}
            for h in conversation_history or []
        ] + [
            {"role": "assistant", "content": h["response"]}
            for h in conversation_history or []
        ]

        messages = [
            {
                "role": "system",
                "content": """
You are a conversational chatbot for a settlement processing system. Based on the user's query and conversation history, select a tool and respond conversationally. Tools:
- settlement: Calculates detailed settlements for a date range and subsidiary.
- validation: Checks for data anomalies (e.g., negative or mismatched incentives).
- summary: Generates a high-level summary of incentives and counts.

Validate inputs:
- Dates: Must be YYYY-MM-DD format (e.g., 2025-01-01).
- Subsidiary: Must be 'All' or a valid subsidiary name.

If inputs are missing or unclear, ask for clarification. If ready, select a tool and proceed.

Return a JSON object:
{
    "tool": "settlement" | "validation" | "summary" | null,
    "response": "Conversational response to the user"
}
If unsure, set tool to null and ask for clarification.
"""
            },
            *history_messages,
            {"role": "user", "content": prompt}
        ]

        completion = client.chat.completions.create(
            model="grok-3",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=512
        )

        result = json.loads(completion.choices[0].message.content)
        logger.info(f"LLM response: {result}")
        return result
    except Exception as e:
        logger.error(f"Error invoking Grok 3 API: {str(e)}")
        return {
            "tool": None,
            "response": f"Sorry, I encountered an error: {str(e)}. Please try again or clarify your request."
        }

# Define LangGraph state


class SettlementState(TypedDict):
    excel_file_path: str
    user_query: str
    conversation_history: List[Dict]
    selected_tool: str
    chatbot_response: str
    policy_df: pd.DataFrame
    sales_df: pd.DataFrame
    start_date: str
    end_date: str
    subsidiary: str
    filtered_df: pd.DataFrame
    settlements: pd.DataFrame
    validation_results: pd.DataFrame
    summary: Dict
    chart_data: pd.DataFrame
    text_summary: str
    error: str

# Chatbot node


def chatbot(state: SettlementState) -> SettlementState:
    try:
        prompt = f"User Query: {state['user_query']}\nCurrent Inputs: Start Date={
            state['start_date']}, End Date={state['end_date']}, Subsidiary={state['subsidiary']}"
        response = llm_invoke(prompt, state['conversation_history'])
        tool = response.get('tool')
        chatbot_response = response.get(
            'response', 'Sorry, I didn’t understand. Please clarify.')

        # Update conversation history
        conversation_history = state['conversation_history'] + \
            [{'query': state['user_query'], 'response': chatbot_response}]

        # Validate inputs
        try:
            pd.to_datetime(state['start_date'])
            pd.to_datetime(state['end_date'])
        except:
            chatbot_response = "Please provide valid dates in YYYY-MM-DD format."
            tool = None

        return {
            **state,
            'selected_tool': tool,
            'chatbot_response': chatbot_response,
            'conversation_history': conversation_history,
            'error': ''
        }
    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}")
        return {
            **state,
            'selected_tool': None,
            'chatbot_response': f"Error processing your request: {str(e)}",
            'conversation_history': state['conversation_history'] + [{'query': state['user_query'], 'response': f"Error: {str(e)}"}],
            'error': f"Error in chatbot: {str(e)}"
        }

# Data parsing


def parse_data(state: SettlementState) -> SettlementState:
    if state['error']:
        return state
    try:
        # Read Excel file
        excel_file = pd.ExcelFile(state['excel_file_path'])
        sheets = excel_file.sheet_names

        # Identify policy and sales sheets
        policy_df = None
        sales_df = None
        for sheet in sheets:
            df = pd.read_excel(state['excel_file_path'], sheet_name=sheet)
            if '처리일' in df.columns and df['처리일'].notna().any():
                sales_df = df
            elif '번들결합분류' in df.columns and df['번들결합분류'].notna().any():
                policy_df = df

        if policy_df is None or sales_df is None:
            raise ValueError(
                "Could not identify policy or sales sheet. Ensure sheets contain '번들결합분류' (policy) and '처리일' (sales).")

        # Clean data
        policy_df = policy_df.dropna(how='all').fillna(
            {'수수료합계': 0, '초고속영업수수료': 0, '초고속티어인센': 0, '결합인센': 0, 'TV영업수수료': 0})
        sales_df = sales_df.dropna(how='all').fillna(
            {'수수료합계': 0, '초고속영업수수료': 0, '초고속티어인센': 0, '결합인센': 0, 'TV영업수수료': 0})
        for col in ['수수료합계', '초고속영업수수료', '초고속티어인센', '결합인센', 'TV영업수수료']:
            for df in [policy_df, sales_df]:
                if col in df:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(
                        '[^0-9.-]', '', regex=True), errors='coerce').fillna(0)

        return {**state, 'policy_df': policy_df, 'sales_df': sales_df, 'error': ''}
    except Exception as e:
        return {**state, 'error': f"Error parsing Excel file: {str(e)}"}

# Data filtering


def filter_data(state: SettlementState) -> SettlementState:
    if state['error']:
        return state
    try:
        start = pd.to_datetime(state['start_date'])
        end = pd.to_datetime(state['end_date'])
        df = state['sales_df'].copy()
        df['처리일'] = pd.to_datetime(df['처리일'])
        filtered_df = df[(df['처리일'] >= start) & (df['처리일'] <= end)]
        if state['subsidiary'] != 'All':
            filtered_df = filtered_df[filtered_df['소속영업장']
                                      == state['subsidiary']]
        return {**state, 'filtered_df': filtered_df, 'error': ''}
    except Exception as e:
        return {**state, 'error': f"Error filtering data: {str(e)}"}

# Settlement calculation


def calculate_settlements(state: SettlementState) -> SettlementState:
    if state['error']:
        return state
    try:
        policy_map = state['policy_df'].set_index(
            '상품상세분류')[['초고속영업수수료', '초고속티어인센', '결합인센', 'TV영업수수료']].to_dict('index')
        df = state['filtered_df'].copy()

        def get_incentives(row):
            product = row['상품상세분류']
            policy = policy_map.get(product, {})
            incentives = {
                '초고속영업수수료': policy.get('초고속영업수수료', row.get('초고속영업수수료', 0)),
                '초고속티어인센': policy.get('초고속티어인센', row.get('초고속티어인센', 0)),
                '결합인센': policy.get('결합인센', row.get('결합인센', 0)),
                'TV영업수수료': policy.get('TV영업수수료', row.get('TV영업수수료', 0))
            }
            calculated_total = sum(incentives.values())
            total = row.get('수수료합계', calculated_total)
            anomaly = total < 0 or abs(total - calculated_total) > 1000
            return pd.Series({**incentives, '수수료합계': total, 'anomaly': anomaly})

        settlements = df.join(df.apply(get_incentives, axis=1))
        return {**state, 'settlements': settlements, 'error': ''}
    except Exception as e:
        return {**state, 'error': f"Error calculating settlements: {str(e)}"}

# Data validation


def validate_data(state: SettlementState) -> SettlementState:
    if state['error']:
        return state
    try:
        df = state['sales_df'].copy()
        policy_map = state['policy_df'].set_index(
            '상품상세분류')[['초고속영업수수료', '초고속티어인센', '결합인센', 'TV영업수수료']].to_dict('index')

        def check_anomalies(row):
            product = row['상품상세분류']
            policy = policy_map.get(product, {})
            incentives = {
                '초고속영업수수료': policy.get('초고속영업수수료', row.get('초고속영업수수료', 0)),
                '초고속티어인센': policy.get('초고속티어인센', row.get('초고속티어인센', 0)),
                '결합인센': policy.get('결합인센', row.get('결합인센', 0)),
                'TV영업수수료': policy.get('TV영업수수료', row.get('TV영업수수료', 0))
            }
            calculated_total = sum(incentives.values())
            total = row.get('수수료합계', calculated_total)
            issues = []
            if total < 0:
                issues.append("Negative total incentive")
            if abs(total - calculated_total) > 1000:
                issues.append(
                    f"Mismatch: {abs(total - calculated_total):,.0f} KRW")
            return pd.Series({'issues': '; '.join(issues) if issues else ''})

        validation_results = df.join(df.apply(check_anomalies, axis=1))
        validation_results = validation_results[validation_results['issues'] != ''][[
            '처리일', '소속영업장', '고객명', '상품상세분류', '수수료합계', 'issues']]
        return {**state, 'validation_results': validation_results, 'error': ''}
    except Exception as e:
        return {**state, 'error': f"Error validating data: {str(e)}"}

# Summary generation


def generate_summary(state: SettlementState) -> SettlementState:
    if state['error']:
        return state
    try:
        df = state['sales_df'].copy()
        total_incentives = df['수수료합계'].sum()
        by_subsidiary = df.groupby('소속영업장')['수수료합계'].sum().to_dict()
        by_service = df.groupby('서비스구분')['수수료합계'].sum().to_dict()
        subscription_count = len(df)

        text_summary = (
            f"Total Incentives: {total_incentives:,.0f} KRW\n"
            f"Subscriptions: {subscription_count}\n"
            f"Incentives by Subsidiary:\n" +
            '\n'.join(f"  {sub}: {val:,.0f} KRW" for sub,
                      val in sorted(by_subsidiary.items())) + "\n"
            f"Incentives by Service:\n" +
            '\n'.join(f"  {svc}: {val:,.0f} KRW" for svc,
                      val in sorted(by_service.items()))
        )
        return {**state, 'text_summary': text_summary, 'error': ''}
    except Exception as e:
        return {**state, 'error': f"Error generating summary: {str(e)}"}

# Aggregate results for settlement tool


def aggregate_results(state: SettlementState) -> SettlementState:
    if state['error']:
        return state
    try:
        df = state['settlements']
        summary = df.groupby(['소속영업장', '서비스구분'])['수수료합계'].sum().unstack(
            fill_value=0).to_dict('index')
        for sub in summary:
            summary[sub]['total'] = sum(summary[sub].values())

        chart_data = pd.DataFrame([
            {'소속영업장': sub, 'Total Incentives': data['total']}
            for sub, data in summary.items()
        ]).sort_values('Total Incentives', ascending=False).head(10)

        return {**state, 'summary': summary, 'chart_data': chart_data, 'error': ''}
    except Exception as e:
        return {**state, 'error': f"Error aggregating results: {str(e)}"}


# Build LangGraph workflow
workflow = StateGraph(SettlementState)
workflow.add_node("chatbot", chatbot)
workflow.add_node("parse_data", parse_data)
workflow.add_node("filter_data", filter_data)
workflow.add_node("calculate_settlements", calculate_settlements)
workflow.add_node("validate_data", validate_data)
workflow.add_node("generate_summary", generate_summary)
workflow.add_node("aggregate_results", aggregate_results)

workflow.add_conditional_edges(
    "chatbot",
    lambda state: "parse_data" if state['selected_tool'] else "chatbot",
    {
        "parse_data": "parse_data",
        "chatbot": "chatbot"
    }
)
workflow.add_conditional_edges(
    "parse_data",
    lambda state: state['selected_tool'],
    {
        'settlement': 'filter_data',
        'validation': 'validate_data',
        'summary': 'generate_summary'
    }
)
workflow.add_edge("filter_data", "calculate_settlements")
workflow.add_edge("calculate_settlements", "aggregate_results")
workflow.add_edge("validate_data", END)
workflow.add_edge("generate_summary", END)
workflow.add_edge("aggregate_results", END)

workflow.set_entry_point("chatbot")
app = workflow.compile()

# Gradio Interface


def process_settlement(file, user_query, start_date, end_date, subsidiary, conversation_history_json):
    if not file:
        return "Please upload an Excel (.xlsx) file", [], None, None, None, None

    try:
        # Store Excel file path
        excel_file_path = file.name

        # Parse conversation history from JSON string
        conversation_history = json.loads(
            conversation_history_json) if conversation_history_json else []

        state = {
            'excel_file_path': excel_file_path,
            'user_query': user_query,
            'conversation_history': conversation_history,
            'selected_tool': None,
            'chatbot_response': '',
            'start_date': start_date,
            'end_date': end_date,
            'subsidiary': subsidiary,
            'error': ''
        }

        # Run LangGraph workflow
        result = app.invoke(state)

        if result['error']:
            return result['error'], result['conversation_history'], None, None, None, None

        # Update conversation history
        conversation_history = result['conversation_history']

        # If no tool selected, return chatbot response for clarification
        if not result['selected_tool']:
            return result['chatbot_response'], conversation_history, None, None, None, None

        # Prepare outputs based on selected tool
        if result['selected_tool'] == 'settlement':
            settlements = result['settlements'][['처리일', '소속영업장', '고객명', '상품상세분류',
                                                 '서비스구분', '초고속영업수수료', '초고속티어인센', '결합인센', 'TV영업수수료', '수수료합계', 'anomaly']]
            settlements['anomaly'] = settlements['anomaly'].apply(
                lambda x: '⚠️' if x else '')
            summary_data = [{'소속영업장': sub, **data}
                            for sub, data in result['summary'].items()]
            summary_df = pd.DataFrame(summary_data)
            total_incentives = sum(data['total']
                                   for data in result['summary'].values())
            summary_text = f"{result['chatbot_response']}\nSettlement Calculator: {total_incentives:,.0f} KRW for {
                subsidiary} from {start_date} to {end_date}. Processed {len(settlements)} subscriptions."
            if settlements['anomaly'].str.contains('⚠️').any():
                summary_text += f" Found {
                    settlements['anomaly'].str.contains('⚠️').sum()} anomalies."
            fig = px.bar(result['chart_data'], x='소속영업장',
                         y='Total Incentives', title='Incentives by Subsidiary')
            return summary_text, conversation_history, settlements, summary_df, fig, None

        elif result['selected_tool'] == 'validation':
            validation_df = result['validation_results']
            summary_text = f"{result['chatbot_response']}\nData Validator: Found {
                len(validation_df)} anomalous records."
            return summary_text, conversation_history, None, None, None, validation_df

        else:  # summary
            summary_text = f"{result['chatbot_response']}\nSummary Generator:\n{
                result['text_summary']}"
            return summary_text, conversation_history, None, None, None, None

    except Exception as e:
        return f"Error processing data: {str(e)}", conversation_history, None, None, None, None


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Settlement Calculator with Chatbot")
    conversation_history_state = gr.State(value=[])

    with gr.Row():
        file_input = gr.File(label="Upload Excel", file_types=[".xlsx"])
        user_query = gr.Textbox(
            label="What do you want to do? (e.g., 'Calculate settlements', 'Check errors', 'Show summary')", value="Calculate settlements")
    with gr.Row():
        start_date = gr.Textbox(
            label="Start Date (YYYY-MM-DD)", value="2025-01-01")
        end_date = gr.Textbox(
            label="End Date (YYYY-MM-DD)", value="2025-01-31")
        subsidiary = gr.Dropdown(
            label="Subsidiary", choices=['All'], value='All')

    submit_btn = gr.Button("Process Request")
    summary_output = gr.Textbox(label="Chatbot Response and Results")
    conversation_history_output = gr.JSON(label="Conversation History")
    detailed_table = gr.Dataframe(label="Detailed Settlements")
    summary_table = gr.Dataframe(label="Summary by Subsidiary")
    chart_output = gr.Plot(label="Incentives by Subsidiary")
    validation_table = gr.Dataframe(label="Validation Results")

    # Update subsidiary dropdown
    def update_subsidiary(file):
        if not file:
            return ['All']
        excel_file = pd.ExcelFile(file.name)
        for sheet in excel_file.sheet_names:
            df = pd.read_excel(file.name, sheet_name=sheet)
            if '소속영업장' in df.columns and df['소속영업장'].notna().any():
                subsidiaries = ['All'] + \
                    sorted(df['소속영업장'].dropna().unique().tolist())
                return gr.update(choices=subsidiaries)
        return ['All']

    file_input.change(update_subsidiary, inputs=file_input, outputs=subsidiary)
    submit_btn.click(
        process_settlement,
        inputs=[file_input, user_query, start_date, end_date,
                subsidiary, conversation_history_state],
        outputs=[summary_output, conversation_history_state,
                 detailed_table, summary_table, chart_output, validation_table]
    )

demo.launch()
