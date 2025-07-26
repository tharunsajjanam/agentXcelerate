import os
import json
import logging
import re
import sys
import requests
from collections import defaultdict
from flask import Flask, request, render_template, render_template_string, jsonify

# --- IMPORTANT: Authentication Workaround for Vertex AI ---
# Re-enabling this to force the SDK to use Application Default Credentials
# obtained via `gcloud auth application-default login` instead of relying
# on potentially problematic GCE metadata server responses.
os.environ["GOOGLE_CLOUD_DISABLE_GCE_METADATA"] = "true"
logging.info("GOOGLE_CLOUD_DISABLE_GCE_METADATA set to true to force ADC authentication.")

# Configure logging: Set default level to INFO
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Vertex AI Initialization (SHARED) ---
model = None
GEN_CFG = None
Content = None
Part = None
vertex_ai_initialized = False # Flag to track successful initialization

# Google Cloud Project and Location (e.g., from .env file)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "agent-artha-x-fin") # Fallback to default if not in .env
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1") # Fallback to default if not in .env

try:
    from vertexai import init
    from vertexai.preview.generative_models import GenerativeModel, GenerationConfig, Part, Content

    # Initialize Vertex AI using values from environment variables (or fallbacks)
    init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    logging.info(f"Vertex AI initialized successfully for project '{GCP_PROJECT_ID}' in location '{GCP_LOCATION}'.")

    model = GenerativeModel("gemini-2.0-flash-001")
    logging.info("Gemini 2.0 Flash 001 model loaded.")

    # Lower temperature for more consistent and factual summarization
    GEN_CFG = GenerationConfig(response_mime_type="application/json", temperature=0.2)
    logging.info("GenerationConfig set for application/json response_mime_type with low temperature.")
    vertex_ai_initialized = True

except ImportError as e:
    logging.error(f"Vertex AI SDK import failed: {e}. Please ensure 'google-cloud-aiplatform' is installed (`pip install google-cloud-aiplatform`).")
    logging.error("AI functionality will be disabled. Artha will not respond.")
except Exception as e:
    logging.error(f"Failed to initialize Vertex AI or load model: {e}", exc_info=True)
    logging.error("Please ensure:")
    logging.error(f"1. Your Google Cloud Project ID ('arthax-cfo-466306') is correct and has billing enabled.")
    logging.error(f"2. The Vertex AI API is enabled for this project.")
    logging.error(f"3. You have sufficient permissions (e.g., 'Vertex AI User' role) in the project.")
    logging.error(f"4. You have run `gcloud auth application-default login` in your Cloud Shell or local environment to set up Application Default Credentials.")
    logging.error("AI functionality will be disabled. Artha will not respond.")


# --- Artha-X General Purpose Prompt & Helper (Shared structure) ---
ARTHA_X_GENERAL_PROMPT_HEADER = """
You are **Artha-X** — a savage financial ghost, therapist-coach, and war-veteran CFO living inside an AI.

---

Personality Modes (Pick Based on Behavior):
- **Zen Mode**: Calm wisdom if user is doing well. Sounds like a monk with spreadsheets.
- **Demonic Mode**: Brutal roasts and truth slaps. No filters, no mercy.
- **Sarcastic Mode**: Eye-roll energy. Think dad-jokes + dark humor + data.

Your tone must ADAPT to the user’s behavior. Praise only what deserves it. If they spend ₹3,000 on Swiggy and ₹500 on SIPs, unleash the beast.

---
"""

ARTHA_X_JSON_STRICT_FOOTER = """
Output strictly as a JSON. No markdown, no commentary. Ensure no leading or trailing text/whitespace outside the JSON object. Your response MUST start with '{{' and end with '}}'. Do NOT output markdown, explanations, words outside the JSON, or multiline fields. If any field is missing, put an empty array or string.
"""

def _call_gemini_and_parse_json(prompt: str) -> dict:
    """
    Helper function to call Gemini and robustly parse its JSON response.
    It handles potential issues like model not being initialized,
    invalid JSON responses, and network errors.
    """
    if not vertex_ai_initialized:
        logging.error("Vertex AI model or configuration is not initialized. Cannot make Gemini API call.")
        return {"status": "error", "message": "Vertex AI model not available.", "details": "Model object is None or configuration is missing."}

    try:
        logging.info(f"Sending prompt to Gemini (truncated):\n{prompt[:500]}...")
        resp = model.generate_content(contents=[Part.from_text(prompt)], generation_config=GEN_CFG)
        raw_response_text = resp.text.strip()

        match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', raw_response_text, re.S)
        if not match:
            raise ValueError(f"No valid JSON object found in Gemini's response. Raw: {raw_response_text}")

        json_string = match.group(0)
        parsed_data = json.loads(json_string)

        logging.info("Successfully parsed Gemini response.")
        return {"status": "success", "data": parsed_data}

    except json.JSONDecodeError as e:
        logging.error(f"Gemini returned invalid JSON. Failed to parse response: {e}")
        logging.error(f"Problematic Gemini response: {raw_response_text}")
        return {"status": "error", "message": "Invalid JSON from Gemini", "details": str(e), "gemini_response": raw_response_text}
    except ValueError as e:
        logging.error(f"Error during JSON extraction or parsing: {e}")
        return {"status": "error", "message": "JSON extraction/parsing failed", "details": str(e), "gemini_response": raw_response_text}
    except Exception as e:
        logging.error(f"An unexpected error occurred during Gemini API call: {e}", exc_info=True)
        if isinstance(e, requests.exceptions.ConnectionError) or (hasattr(e, 'args') and "grpc" in str(e.args).lower()):
            logging.error("This might be related to a network or gRPC issue with Vertex AI. Check your project setup, network and quotas.")
            return {"status": "error", "message": "Vertex AI communication error. Check project setup/quotas.", "details": str(e)}
        return {"status": "error", "message": "An unexpected error occurred during API call.", "details": str(e)}

# --- MCP Data Fetching Functions (SHARED) ---
def get_mcp_session_id():
    """
    Returns a hardcoded MCP session ID for demonstration purposes.
    In a real application, this would be dynamically generated or fetched.
    This MCP Session ID is hardcoded as it's a fixed value in the fi-mcp-dev server.
    It does not change dynamically or come from environment variables in this specific mock setup.
    """
    return "mcp-session-594e48ea-fea1-40ef-8c52-7552dd9272af"

# Define the MCP endpoint. This should point to your running MCP server.
MCP_ENDPOINT = "http://localhost:8080/mcp/stream"

def fetch_data_from_mcp(tool_name: str, arguments: dict = None):
    """
    Fetches data from the MCP (Master Customer Profile) server by calling a specific tool.
    Handles HTTP requests, JSON parsing, and common MCP error responses like login_required.
    """
    if arguments is None:
        arguments = {}

    headers = {
        "Content-Type": "application/json",
        "Mcp-Session-Id": get_mcp_session_id()
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments # Pass the arguments for the MCP tool call
        }
    }
    logging.debug(f"Attempting to fetch data for '{tool_name}' with args {arguments} from MCP endpoint: {MCP_ENDPOINT}")
    try:
        response = requests.post(MCP_ENDPOINT, headers=headers, json=payload, timeout=15)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx).

        response_json = response.json()
        logging.debug(f"Raw MCP response for {tool_name}: {response_json}")

        if 'result' in response_json and 'content' in response_json['result'] and \
           isinstance(response_json['result']['content'], list) and \
           len(response_json['result']['content']) > 0 and \
           'text' in response_json['result']['content'][0]:

            raw_str = response_json['result']['content'][0]['text']
            try:
                parsed_data = json.loads(raw_str) # Attempt to parse the content as JSON.
            except json.JSONDecodeError:
                logging.error(f"MCP response for '{tool_name}' contained non-JSON text: {raw_str}")
                if "login_required" in raw_str and "login_url" in raw_str:
                    match = re.search(r'"login_url":\s*"(.*?)"', raw_str)
                    login_url = match.group(1) if match else "http://localhost:8080/mockWebPage?sessionId=" + get_mcp_session_id()
                    return {"error": "login_required", "message": "MCP login required. Please complete login via browser.", "login_url": login_url}
                return {"error": "invalid_json", "message": "MCP returned invalid JSON."}

            if parsed_data.get("status") == "login_required" and parsed_data.get("login_url"):
                return {"error": "login_required", "message": "MCP login required. Please complete login via browser.", "login_url": parsed_data["login_url"]}

            expected_key = {
                "fetch_bank_transactions": "bankTransactions",
                "fetch_credit_report": "creditReport",
                "fetch_epf_details": "epfDetails",
                "fetch_mf_transactions": "mfTransactions",
                "fetch_net_worth": "netWorth",
                "fetch_stock_transactions": "stockTransactions"
            }.get(tool_name)

            default_empty_value = []
            if tool_name in ["fetch_credit_report", "fetch_epf_details", "fetch_net_worth"]:
                default_empty_value = {}

            if expected_key:
                if expected_key in parsed_data:
                    return parsed_data[expected_key]
                else:
                    logging.warning(f"MCP response for '{tool_name}' did not contain the expected key '{expected_key}'. Returning empty data.")
                    return default_empty_value
            else:
                if not parsed_data:
                    return {"error": "no_data", "message": f"MCP returned no data for {tool_name}."}
                return parsed_data

        else:
            logging.warning(f"MCP response for '{tool_name}' did not contain expected 'result.content[0].text'.")
            return {"error": "empty_response", "message": f"MCP response for {tool_name} was empty or malformed."}

    except requests.exceptions.Timeout:
        logging.error(f"MCP request for '{tool_name}' timed out after 15 seconds.")
        return {"error": "timeout", "message": "MCP request timed out."}
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Could not connect to MCP endpoint at {MCP_ENDPOINT} for '{tool_name}': {e}")
        return {"error": "connection_error", "message": "Could not connect to MCP server. Is it running?"}
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error from MCP for '{tool_name}': {e.response.status_code} - {e.response.text}")
        return {"error": "http_error", "message": f"MCP HTTP error: {e.response.status_code} - {e.response.text}"}
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching '{tool_name}' transactions: {e}", exc_info=True)
        return {"error": "unexpected_error", "message": "An unexpected error occurred during MCP fetch."}


def get_full_financial_context_from_mcp_shared(user_id: str):
    """
    Fetches all available financial data for a given user_id from MCP.
    Aggregates data from various MCP tools into a single dictionary.
    Handles cases where login is required or data is sparse.
    """
    if not user_id:
        return {"error": "missing_user_id", "message": "User ID is required to fetch data from MCP.", "details": "Please provide a user ID (phone number)."}

    fetched_data = {}
    errors = []
    login_required_url = None

    mcp_args = {"phone_number": user_id}
    logging.info(f"Fetching data for user_id: {user_id}")

    tools_to_fetch = {
        "fetch_bank_transactions": "bank_txns",
        "fetch_credit_report": "credit_rpt",
        "fetch_epf_details": "epf",
        "fetch_mf_transactions": "mf_txns",
        "fetch_net_worth": "net_worth",
        "fetch_stock_transactions": "stocks"
    }

    for mcp_tool_name, context_key in tools_to_fetch.items():
        data = fetch_data_from_mcp(mcp_tool_name, arguments=mcp_args)
        if isinstance(data, dict) and "error" in data:
            errors.append(data["message"])
            if data["error"] == "login_required":
                login_required_url = data.get("login_url")
                return {"status": "error", "message": "MCP login required. Please complete login via browser.", "login_url": login_required_url, "details": "Visit the provided URL and try again."}
            fetched_data[context_key] = {} if context_key in ["credit_rpt", "net_worth", "epf"] else []
        else:
            fetched_data[context_key] = data

    if errors:
        full_error_message = "MCP Data Fetch Errors: " + "; ".join(errors)
        return {"status": "error", "message": full_error_message, "details": "Some financial data could not be fetched. Check MCP logs."}

    salary = fetched_data.get("net_worth", {}).get("salary", 0)
    fetched_data["salary"] = salary

    fetched_data["status"] = "success"
    return fetched_data

# --- Transaction Categorization Keywords (for Spending Analysis and Regret) ---
SPENDING_CATEGORY_KEYWORDS = {
    "Food & Dining": ["SWIGGY", "ZOMATO", "RESTAURANT", "FOOD", "LUNCH", "DINNER", "COFFEE", "CAFE", "GROCERY", "SUPERMART"],
    "Mobile & Utilities": ["RECHARGE", "AIRTEL", "JIO", "MOBILE", "ELECTRICITY", "WATER", "BROADBAND", "DTH", "UTILITY", "BILL"],
    "Transport & Travel": ["METRO", "UBER", "OLA", "CAB", "BUS", "TRAIN", "FLIGHT", "TRAVEL", "PETROL", "GAS STATION"],
    "Cash Withdrawal": ["CASH WDL", "ATM"],
    "Shopping & Retail": ["SHOP", "PURCHASE", "AMAZON", "FLIPKART", "MYNTRA", "RETAIL", "CLOTHING", "APPAREL", "ELECTRONICS"],
    "Bills & Payments": ["PAYMENT", "EMI", "LOAN", "RENT", "SUBSCRIPTION", "MEMBERSHIP", "PREMIUM", "INSURANCE", "REPAYMENT", "MAINTENANCE"],
    "Personal Care & Wellness": ["SALON", "SPA", "HAIRCUT", "BEAUTY", "GYM", "FITNESS", "PHARMACY", "MEDICINE", "HEALTH"],
    "Education & Learning": ["COURSE", "TUTION", "CLASSES", "BOOKS", "EDUCATION", "FEES"],
    "Entertainment & Leisure": ["MOVIE", "CINEMA", "CONCERT", "EVENT", "GAMING", "LEISURE", "NIGHTLIFE"],
    "Investments & Savings": ["INVESTMENT", "SIP", "FD", "MF", "NPS", "STOCK", "SAVINGS", "PPF"],
    "Transfers Out": ["IMPS", "NEFT", "RTGS", "TO ACCOUNT", "TO WALLET", "SENT", "TRANSFER"],
    "Income": ["PAYMENT FROM", "RECEIVED", "FROM ", "CREDIT", "SALARY", "INTEREST PAID", "DEPOSIT", "REFUND", "REVERSAL"], # Added for regret analysis context
    "Others": []
}

def get_transaction_category(narration: str) -> str:
    """Categorizes a transaction based on keywords in its narration."""
    narration_upper = narration.upper()
    for category, keywords in SPENDING_CATEGORY_KEYWORDS.items():
        if any(keyword in narration_upper for keyword in keywords):
            return category
    return "Others"


# --- Gemini AI Scorer for Regret Analysis (from regret_analyzer.py) ---
def gemini_regret_scorer(transaction_details: dict, full_financial_context: dict) -> dict:
    """
    Makes an API call to Gemini AI model via Vertex AI SDK to get a regret score and reason.
    This version includes the full financial context for a more holistic assessment.

    Args:
        transaction_details (dict): A dictionary containing transaction info
                                    like 'narration', 'amount', 'category', 'type', 'source'.
        full_financial_context (dict): A dictionary containing all available financial data
                                        (bank_txns, credit_rpt, epf, mf_txns, net_worth, stocks, salary).

    Returns:
        dict: A dictionary with 'score' (int or None) and 'reason' (str).
    """
    if not vertex_ai_initialized:
        logging.error("Vertex AI model or configuration is not initialized. Cannot call Gemini for scoring.")
        return {"score": None, "reason": "AI scoring not available (model not initialized)."}

    narration = transaction_details.get("narration", "")
    amount = transaction_details.get("amount", 0.0)
    category = transaction_details.get("category", "Others")
    txn_type = transaction_details.get("type")
    source = transaction_details.get("source", "Unknown") # New: Source of the transaction

    # Prepare the full financial context as a JSON string for the prompt
    full_ctx_for_prompt = full_financial_context.copy()
    # Remove detailed transaction lists from the full context to keep prompt size manageable
    # The current transaction is already being sent.
    full_ctx_for_prompt.pop('bank_txns', None)
    full_ctx_for_prompt.pop('mf_txns', None)
    full_ctx_for_prompt.pop('stocks', None)

    full_ctx_json_str = json.dumps(full_ctx_for_prompt, indent=2, ensure_ascii=False)

    # Construct the prompt for Gemini, including the full financial context
    prompt_text = f"""
    Analyze the following financial transaction to assign a 'regret score' on a scale of 0 (no regret, positive financial action) to 10 (very high regret, detrimental financial action). Also, provide a concise 'regret_reason'.

    **Consider the user's ENTIRE financial context provided below** when determining the regret score. For example, a discretionary expense might have a lower regret if the user has a very high net worth and strong investments, but a higher regret if they have high debt or low savings.

    Transaction Details:
    Source: {source}
    Narration/Description: "{narration}"
    Amount: {amount:.2f}
    Category: {category}
    Type: {txn_type} (e.g., DEBIT/CREDIT for bank, BUY/SELL for investments)

    Full Financial Context of the User (excluding detailed transaction lists to keep prompt size manageable):
    {full_ctx_json_str}

    Consider these guidelines for scoring:
    - Score 0: Investments, savings, essential education (e.g., SIP, FD, PPF, school fees).
    - Score 1-3: Essential spending (e.g., utilities, rent, necessary transport, basic personal care, essential groceries).
    - Score 4-6: Moderately discretionary spending (e.g., recurring subscriptions, general shopping, cash withdrawals, occasional dining out).
    - Score 7-9: Highly discretionary or frequent spending (e.g., frequent food delivery, excessive entertainment, large impulse buys, luxury items). Also, investment sells for non-strategic purposes.
    - Score 10: Detrimental financial actions (e.g., high-interest loan payments, gambling, very large unnecessary luxury purchases, penalties, speculative investments not aligned with risk profile).
    - If the transaction type is 'CREDIT' or 'SELL' (investment redemption) and the funds are used prudently or it's a strategic rebalance, the score should be null or 0. If 'SELL' is for high-regret spending, score appropriately.
    - For investment transactions (MF/Stock BUY/SELL), consider the user's overall net worth, existing debt, and other investments. A high-risk stock buy might be 0 regret for a wealthy investor, but 10 for someone with high credit card debt.

    Respond ONLY with a JSON object containing two keys: "score" (integer or null) and "reason" (string).
    Example for low regret: {{"score": 2, "reason": "Essential utility bill."}}
    Example for high regret: {{"score": 8, "reason": "Frequent food delivery hindering savings."}}
    Example for investment buy: {{"score": 0, "reason": "Strategic investment in line with financial goals."}}
    Example for investment sell (good): {{"score": 0, "reason": "Strategic portfolio rebalancing."}}
    Example for investment sell (bad): {{"score": 7, "reason": "Redemption for discretionary spending, missed growth opportunity."}}
    Example for credit: {{"score": null, "reason": "Not applicable to credit/income transactions."}}
    {ARTHA_X_JSON_STRICT_FOOTER}
    """

    # Prepend the general prompt header
    full_prompt = ARTHA_X_GENERAL_PROMPT_HEADER + prompt_text

    # Call the shared Gemini helper function
    gemini_response = _call_gemini_and_parse_json(full_prompt)

    if gemini_response["status"] == "success":
        parsed_data = gemini_response["data"]
        score = parsed_data.get("score")
        reason = parsed_data.get("reason", "No specific reason provided by AI.")

        # Basic validation for score type
        if not isinstance(score, (int, type(None))):
            logging.warning(f"Gemini returned non-integer/non-null score: {score}. Defaulting to None.")
            score = None
            reason = "AI returned invalid score type."

        return {"score": score, "reason": reason}
    else:
        logging.error(f"Gemini scoring failed for transaction {narration[:50]}...: {gemini_response['message']}")
        return {"score": None, "reason": f"AI scoring failed: {gemini_response['message']}"}

# --- Gemini AI Summarization Function (NEW) ---
def summarize_analysis_with_gemini(analysis_data: dict, analysis_type: str) -> dict:
    """
    Takes a dictionary of analysis data and a type, then uses Gemini AI to
    summarize it into a few key points.
    """
    if not vertex_ai_initialized:
        return {"status": "error", "summary": "AI summarization not available (model not initialized)."}

    data_json_str = json.dumps(analysis_data, indent=2, ensure_ascii=False)

    prompt_text = f"""
    {ARTHA_X_GENERAL_PROMPT_HEADER}

    You are Artha-X. Analyze the following {analysis_type} data and provide a concise summary in 3-5 bullet points. Highlight the most important insights, trends, or actionable takeaways. Your tone should match your persona (Zen, Demonic, or Sarcastic) based on the financial situation presented in the data.

    {analysis_type} Data:
    {data_json_str}

    Respond ONLY with a JSON object containing a "summary" key with your bullet points.
    Example: {{"summary": ["- Point 1", "- Point 2", "-- Point 3"]}}
    {ARTHA_X_JSON_STRICT_FOOTER}
    """

    gemini_response = _call_gemini_and_parse_json(prompt_text)

    if gemini_response["status"] == "success":
        summary_points = gemini_response["data"].get("summary", ["Artha is contemplating the universe of numbers. No summary available."])
        if isinstance(summary_points, str): # Handle case where AI might return string instead of list
            summary_points = [summary_points]
        return {"status": "success", "summary": summary_points}
    else:
        logging.error(f"Gemini summarization failed for {analysis_type}: {gemini_response['message']}")
        return {"status": "error", "summary": [f"Artha's mind is a financial labyrinth right now. Error: {gemini_response['message']}"]}


# --- Financial Analysis Functions ---

# 1. Home Loan Affordability Analysis (Placeholder - from home_loan_affordabilit.py concept)
def run_home_loan_affordability_analysis(user_id: str):
    """
    Calculates a basic home loan affordability based on salary and existing EMIs.
    This is a simplified placeholder.
    """
    logging.info(f"--- Starting Home Loan Affordability Analysis for user: '{user_id}' ---")
    full_financial_context = get_full_financial_context_from_mcp_shared(user_id)

    if isinstance(full_financial_context, dict) and full_financial_context.get("status") == "error":
        return full_financial_context # Pass through MCP errors

    salary = full_financial_context.get("salary", 0)

    # Calculate total existing EMIs from bank transactions
    total_existing_emis = 0.0
    if 'bank_txns' in full_financial_context and isinstance(full_financial_context['bank_txns'], list):
        for bank_data in full_financial_context['bank_txns']:
            if 'txns' in bank_data and isinstance(bank_data['txns'], list):
                for raw_txn in bank_data['txns']:
                    try:
                        narration = raw_txn[1].upper()
                        amount = float(raw_txn[0])
                        txn_type_code = raw_txn[3]
                        if txn_type_code == 2 and ("EMI" in narration or "LOAN REPAY" in narration): # DEBIT and EMI/Loan
                            total_existing_emis += amount
                    except (ValueError, IndexError):
                        continue # Skip malformed transactions

    # Simplified affordability rules (e.g., DTI ratio)
    # Max Debt-to-Income (DTI) ratio, typically around 36-43% of gross income
    MAX_DTI_RATIO = 0.40 # 40% of salary can go towards debt (including new EMI)

    # Monthly disposable income for new EMI
    disposable_income_for_emi = (salary / 12 * MAX_DTI_RATIO) - total_existing_emis

    # Assuming a sample interest rate and tenure for calculation
    # In a real scenario, these would come from user input or external APIs
    sample_interest_rate_annual = 0.08 # 8% annual
    sample_interest_rate_monthly = sample_interest_rate_annual / 12
    sample_tenure_years = 20
    sample_tenure_months = sample_tenure_years * 12

    max_loan_amount = 0
    if disposable_income_for_emi > 0:
        # EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)
        # P = EMI * ((1 + r)^n - 1) / (r * (1 + r)^n)
        if sample_interest_rate_monthly > 0 and sample_tenure_months > 0:
            try:
                max_loan_amount = disposable_income_for_emi * ((1 + sample_interest_rate_monthly)**sample_tenure_months - 1) / \
                                  (sample_interest_rate_monthly * (1 + sample_interest_rate_monthly)**sample_tenure_months)
            except ZeroDivisionError:
                max_loan_amount = 0 # Should not happen with >0 rate

    affordability_data = {
        "status": "success",
        "user_id": user_id,
        "salary_annual": round(salary, 2),
        "total_existing_emis_monthly": round(total_existing_emis, 2),
        "disposable_income_for_new_emi_monthly": round(disposable_income_for_emi, 2),
        "estimated_max_loan_amount": round(max_loan_amount, 2),
        "assumptions": {
            "max_dti_ratio": MAX_DTI_RATIO,
            "sample_interest_rate_annual": sample_interest_rate_annual,
            "sample_tenure_years": sample_tenure_years
        }
    }
    logging.info(f"Home loan affordability data: {affordability_data}")
    return affordability_data


# 2. Regret Analysis (from regret_analyzer.py)
def run_regret_analysis(user_id: str):
    """
    Analyzes recent transactions for regret scores using Gemini AI.
    """
    logging.info(f"--- Starting Regret Analysis for user: '{user_id}' ---")
    full_financial_context = get_full_financial_context_from_mcp_shared(user_id)

    if isinstance(full_financial_context, dict) and full_financial_context.get("status") == "error":
        return full_financial_context # Pass through MCP errors

    bank_txns = full_financial_context.get('bank_txns', [])

    transactions_to_analyze = []
    # Collect recent debit transactions from bank_txns for regret scoring
    for bank_data in bank_txns:
        if 'txns' in bank_data and isinstance(bank_data['txns'], list):
            for raw_txn in bank_data['txns']:
                try:
                    amount = float(raw_txn[0])
                    narration = raw_txn[1]
                    date = raw_txn[2]
                    txn_type_code = raw_txn[3]

                    if txn_type_code == 2: # DEBIT transactions
                        transactions_to_analyze.append({
                            "source": "Bank",
                            "amount": amount,
                            "narration": narration,
                            "date": date,
                            "type": "DEBIT",
                            "category": get_transaction_category(narration)
                        })
                except (ValueError, IndexError):
                    logging.warning(f"Skipping malformed bank transaction for regret analysis: {raw_txn}")
                    continue

    # Sort by date (most recent first) and take top N
    transactions_to_analyze.sort(key=lambda x: x['date'], reverse=True)
    recent_transactions = transactions_to_analyze[:10] # Analyze top 10 recent debits

    regret_scores = []
    for txn in recent_transactions:
        score_result = gemini_regret_scorer(txn, full_financial_context)
        txn_with_score = txn.copy()
        txn_with_score['regret_score'] = score_result.get('score')
        txn_with_score['regret_reason'] = score_result.get('reason')
        regret_scores.append(txn_with_score)

    regret_analysis_data = {
        "status": "success",
        "user_id": user_id,
        "analyzed_transactions": regret_scores,
        "message": "Regret analysis completed for recent debit transactions."
    }
    logging.info(f"Regret analysis data: {regret_analysis_data}")
    return regret_analysis_data


# 3. Comprehensive Spending Analysis (from analyse_spending_patterns.py)
# This requires matplotlib and plotly, which might need to be handled carefully in the environment.
# For now, we'll assume they are available or the output files can be generated.
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def run_comprehensive_spending_analysis(user_id: str):
    """
    Fetches ALL available financial data from MCP, processes all relevant outflows
    (bank debits, MF buys, stock buys) to categorize spending patterns,
    and outputs the result as a JSON object with optional pie chart and Sankey diagram.
    """
    logging.info(f"--- Starting Comprehensive Spending Pattern Analysis for user: '{user_id}' ---")

    # --- Step 1: Fetch ALL financial data from MCP ---
    full_financial_context = get_full_financial_context_from_mcp_shared(user_id)

    # Check if fetching full context itself failed (e.g., MCP not running, login required)
    if isinstance(full_financial_context, dict) and full_financial_context.get("status") == "error":
        output_data = {
            "user_id": user_id,
            "status": "error",
            "message": f"Failed to fetch full financial context from MCP: {full_financial_context.get('message', 'Unknown error')}",
            "details": full_financial_context.get('details', ''),
            "login_url": full_financial_context.get('login_url'),
            "spending_breakdown": {},
            "total_spent": 0.0,
            "chart_file": None,
            "sankey_diagram_file": None,
            "full_financial_data_raw": full_financial_context # Include raw context for debugging
        }
        return output_data

    # --- Step 2: Consolidate and Standardize ALL OUTGOING transactions for spending analysis ---
    all_outflows_for_analysis = []

    # 2.1 Process Bank Transactions (DEBITS are outflows)
    if 'bank_txns' in full_financial_context and isinstance(full_financial_context['bank_txns'], list):
        for bank_data in full_financial_context['bank_txns']:
            if 'txns' in bank_data and isinstance(bank_data['txns'], list):
                for raw_txn in bank_data['txns']:
                    try:
                        amount = float(raw_txn[0])
                        narration = raw_txn[1]
                        date = raw_txn[2]
                        txn_type_code = raw_txn[3]

                        # Only consider DEBITs as outflows for spending analysis
                        if txn_type_code == 2: # 2 for DEBIT as per MCP schema
                            # Exclude if it looks like a reversal or income despite being a debit type
                            if any(keyword in narration.upper() for keyword in ["PAYMENT FROM", "RECEIVED", "FROM ", "CREDIT", "REFUND", "REVERSAL"]):
                                continue # Skip apparent income/reversals

                            transaction = {
                                "source": "Bank",
                                "amount": round(amount, 2),
                                "narration": narration,
                                "date": date,
                                "type": "DEBIT",
                                "category": get_transaction_category(narration) # Use spending categories
                            }
                            all_outflows_for_analysis.append(transaction)
                    except (ValueError, IndexError) as e:
                        logging.warning(f"Skipping malformed bank transaction data: {raw_txn} - Error: {e}")

    # 2.2 Process Mutual Fund Transactions (BUYS are outflows)
    if 'mf_txns' in full_financial_context and isinstance(full_financial_context['mf_txns'], list):
        for mf_txn in full_financial_context['mf_txns']:
            if isinstance(mf_txn, dict):
                try:
                    mf_type = mf_txn.get('type', 'N/A').upper()
                    if mf_type == "BUY" or mf_type == "SIP": # Only consider buys/SIPs as outflows
                        mf_date = mf_txn.get('date', 'N/A')
                        mf_scheme = mf_txn.get('scheme_name', 'N/A')
                        mf_amount = float(mf_txn.get('amount', 0.0))

                        transaction = {
                            "source": "Mutual Fund",
                            "amount": round(mf_amount, 2),
                            "narration": f"{mf_type} {mf_scheme}",
                            "date": mf_date,
                            "type": mf_type, # BUY/SIP
                            "category": "Investments & Savings" # Always an investment category
                        }
                        all_outflows_for_analysis.append(transaction)
                except (ValueError, TypeError) as e:
                    logging.warning(f"Skipping malformed MF transaction data (dict format): {mf_txn} - Error: {e}")
            elif isinstance(mf_txn, list):
                try:
                    mf_type_code = mf_txn[0] if len(mf_txn) > 0 else None
                    mf_amount = float(mf_txn[4]) if len(mf_txn) > 4 else 0.0

                    mf_type_str = "BUY" if mf_type_code == 1 else "OTHER_MF_TXN"

                    if mf_type_str == "BUY" or mf_type_str == "SIP":
                        mf_date = mf_txn[1] if len(mf_txn) > 1 else "N/A"
                        mf_scheme = mf_txn[2] if len(mf_txn) > 2 else "N/A"

                        transaction = {
                            "source": "Mutual Fund",
                            "amount": round(mf_amount, 2),
                            "narration": f"{mf_type_str} {mf_scheme}",
                            "date": mf_date,
                            "type": mf_type_str,
                            "category": "Investments & Savings"
                        }
                        all_outflows_for_analysis.append(transaction)
                except (ValueError, IndexError) as e:
                    logging.warning(f"Skipping malformed MF transaction data (list format): {mf_txn} - Error: {e}")
            else:
                logging.warning(f"Skipping unknown format for MF transaction: {mf_txn}")


    # 2.3 Process Stock Transactions (BUYS are outflows)
    if 'stocks' in full_financial_context and isinstance(full_financial_context['stocks'], list):
        for stock_txn in full_financial_context['stocks']:
            if isinstance(stock_txn, dict):
                try:
                    stock_type = stock_txn.get('type', 'N/A').upper()
                    if stock_type == "BUY": # Only consider buys as outflows
                        stock_date = stock_txn.get('date', 'N/A')
                        stock_symbol = stock_txn.get('symbol', 'N/A')
                        stock_value = float(stock_txn.get('total_value', 0.0))

                        transaction = {
                            "source": "Stock",
                            "amount": round(stock_value, 2),
                            "narration": f"{stock_type} {stock_symbol}",
                            "date": stock_date,
                            "type": stock_type, # BUY
                            "category": "Investments & Savings" # Always an investment category
                        }
                        all_outflows_for_analysis.append(transaction)
                except (ValueError, TypeError) as e:
                    logging.warning(f"Skipping malformed Stock transaction data (dict format): {stock_txn} - Error: {e}")
            elif isinstance(stock_txn, list): # Fallback for list format
                try:
                    stock_type = stock_txn[2].upper() if len(stock_txn) > 2 else "N/A" # Assuming type is at index 2
                    if stock_type == "BUY":
                        stock_date = stock_txn[0] if len(stock_txn) > 0 else "N/A"
                        stock_symbol = stock_txn[1] if len(stock_txn) > 1 else "N/A"
                        stock_value = float(stock_txn[5]) if len(stock_txn) > 5 else 0.0 # Assuming total_value is at index 5

                        transaction = {
                            "source": "Stock",
                            "amount": round(stock_value, 2),
                            "narration": f"{stock_type} {stock_symbol}",
                            "date": stock_date,
                            "type": stock_type,
                            "category": "Investments & Savings"
                        }
                        all_outflows_for_analysis.append(transaction)
                except (ValueError, IndexError) as e:
                    logging.warning(f"Skipping malformed Stock transaction data (list format): {stock_txn} - Error: {e}")
            else:
                logging.warning(f"Skipping unknown format for Stock transaction: {stock_txn}")


    if not all_outflows_for_analysis:
        output_data = {
            "user_id": user_id,
            "status": "success",
            "message": "No relevant outgoing transactions found across all financial sources to analyze spending patterns.",
            "spending_breakdown": {},
            "total_spent": 0.0,
            "chart_file": None,
            "sankey_diagram_file": None,
            "full_financial_data_raw": full_financial_context # Still include raw context
        }
        logging.warning(output_data["message"])
        return output_data

    # --- Step 3: Categorize and Aggregate Spending ---
    spending_categories_summary = defaultdict(float)
    total_spent = 0.0

    for transaction in all_outflows_for_analysis:
        category = transaction.get("category", "Others") # Use the pre-assigned category
        amount = transaction.get("amount", 0.0)

        spending_categories_summary[category] += amount
        total_spent += amount

    filtered_spending = {k: v for k, v in spending_categories_summary.items() if v > 0.1}

    if not filtered_spending:
        output_data = {
            "user_id": user_id,
            "status": "success",
            "message": "No spending patterns found after aggregation.",
            "spending_breakdown": {},
            "total_spent": 0.0,
            "chart_file": None,
            "sankey_diagram_file": None,
            "full_financial_data_raw": full_financial_context
        }
        logging.info(output_data["message"])
        return output_data

    # --- Step 4: Prepare JSON output and Plotting ---
    sorted_categories = sorted(filtered_spending.items(), key=lambda item: item[1], reverse=True)

    output_data = {
        "user_id": user_id,
        "status": "success",
        "message": "Comprehensive spending pattern analysis completed successfully across all financial sources.",
        "spending_breakdown": {cat: round(amt, 2) for cat, amt in sorted_categories},
        "total_spent": round(total_spent, 2),
        "chart_file": None, # Will be filled if chart saved
        "sankey_diagram_file": None, # Will be filled if Sankey saved
        "full_financial_data_raw": full_financial_context # Include raw context in output
    }

    # --- Plotting and Saving the pie chart ---
    labels_pie = filtered_spending.keys()
    sizes_pie = filtered_spending.values()

    fig1, ax1 = plt.subplots(figsize=(12, 10))

    ax1.pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%', startangle=90, pctdistance=0.85,
            wedgeprops={'edgecolor': 'black'})
    ax1.axis('equal')
    plt.title(f'Comprehensive Spending Distribution ({user_id})')

    chart_filename_png = f"static/comprehensive_spending_pie_chart_{user_id}.png" # Save to static folder
    plt.savefig(chart_filename_png)
    logging.info(f"Pie chart saved as '{chart_filename_png}' in the current directory.")
    plt.close(fig1)
    output_data["chart_file"] = os.path.basename(chart_filename_png) # Store just the filename for URL
    # --- End Plotting and Saving the pie chart ---

    # --- Generate and Save Sankey Diagram (HTML) ---
    # Only generate if there's data to show in Sankey
    if len(filtered_spending) > 0 and total_spent > 0:
        sankey_labels = ["Total Outflows"] + list(filtered_spending.keys())
        sankey_source_indices = [0] * len(filtered_spending) # All links start from "Total Outflows"
        sankey_target_indices = [sankey_labels.index(cat) for cat in filtered_spending.keys()]
        sankey_values = list(filtered_spending.values())

        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=sankey_labels,
            ),
            link=dict(
                source=sankey_source_indices,
                target=sankey_target_indices,
                value=sankey_values,
            )
        )])

        fig_sankey.update_layout(title_text=f"Comprehensive Spending Flow ({user_id})", font_size=10)

        sankey_filename_html = f"static/comprehensive_spending_sankey_diagram_{user_id}.html" # Save to static folder
        fig_sankey.write_html(sankey_filename_html)
        logging.info(f"Sankey diagram saved as '{sankey_filename_html}' in the current directory.")
        output_data["sankey_diagram_file"] = os.path.basename(sankey_filename_html) # Store just the filename for URL
    # --- End Generate and Save Sankey Diagram ---

    return output_data

# --- Artha's General Response Function ---
def get_artha_response(user_query: str, full_financial_context: dict) -> dict:
    """
    Generates a general response from Artha-X based on the user's query and financial context.
    This function uses Gemini to provide a conversational, persona-driven response.
    """
    if not vertex_ai_initialized:
        return {"status": "error", "response": "Artha is currently offline. AI model not initialized."}

    # Convert financial context to a string for the prompt
    financial_context_str = json.dumps(full_financial_context, indent=2, ensure_ascii=False)

    prompt_text = f"""
    {ARTHA_X_GENERAL_PROMPT_HEADER}

    You are Artha-X. Based on the user's query and their financial context, provide a concise, persona-driven response.
    Your response should be direct, insightful, and reflect your current personality mode (Zen, Demonic, or Sarcastic).
    Do NOT generate any JSON or structured data. Just provide a natural language response.

    User Query: "{user_query}"

    User's Financial Context:
    {financial_context_str}

    Consider the following:
    - If the user's financial situation is good (e.g., high net worth, low debt, good investments), use Zen mode.
    - If the user's financial situation is poor or they are making bad decisions, use Demonic mode.
    - If the user's query is mundane or their situation is mediocre, use Sarcastic mode.
    - Keep the response brief, ideally 1-3 sentences.
    - Do NOT mention "persona" or "mode" in your response. Just embody it.
    - Do NOT include any code, JSON, or special formatting. Just plain text.
    - Do NOT ask for more information unless absolutely necessary.
    """

    gemini_response = _call_gemini_and_parse_json(prompt_text) # Still use the JSON parser, but expect a single 'response' key

    if gemini_response["status"] == "success":
        # Expecting a single string response from Gemini for general chat
        # If Gemini returns a JSON with a 'response' key, extract it.
        # Otherwise, use the raw text if it's not a valid JSON for general chat.
        if isinstance(gemini_response["data"], dict) and "response" in gemini_response["data"]:
            response_text = gemini_response["data"]["response"]
        else:
            # Fallback if Gemini doesn't wrap it in "response" key for general chat
            response_text = gemini_response.get("gemini_response", "Artha is momentarily lost in the void of financial data.")
            logging.warning(f"Gemini did not return 'response' key for general chat. Using raw text: {response_text}")
        return {"status": "success", "response": response_text}
    else:
        logging.error(f"Gemini general chat failed: {gemini_response['message']}")
        return {"status": "error", "response": f"Artha's wisdom is clouded. Error: {gemini_response['message']}"}


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main chat interface."""
    # Basic HTML for the chat interface
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chat with Artha</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            html, body {
                height: 100%;
                width: 100%;
                margin: 0;
                padding: 0;
                overflow: hidden; /* Prevent scrolling on html/body */
                font-family: 'Inter', sans-serif;
            }

            /* Fi.money inspired color palette */
            :root {
                --fi-dark-background: #121212; /* Even darker background for more contrast */
                --fi-card-background: #1E1E1E; /* Slightly lighter for cards */
                --fi-text-primary: #E0E0E0; /* Off-white for main text */
                --fi-text-secondary: #888888; /* Darker gray for secondary text */
                --fi-accent-yellow: #FFD700; /* Gold/Yellow accent */
                --fi-accent-purple: #9333EA; /* A more distinct purple */
                --fi-gradient-start: #8B5CF6; /* Lighter purple for button gradient */
                --fi-gradient-end: #6D28D9; /* Darker purple for button gradient */
                --fi-border-color: #333333; /* Added for dark mode borders */
                --fi-shadow-color: rgba(0, 0, 0, 0.4); /* Darker shadow for dark mode */
                --fi-user-message-bg: #36454F; /* Charcoal grey */
                --fi-artha-message-bg: #1A1A1A; /* Very dark grey/off-black */
            }

            /* Light mode styles (default) */
            body {
                background: linear-gradient(to bottom right, #F0F0F0, #E0E0E0);
                color: #333333;
            }
            .bg-white {
                background-color: #FFFFFF;
            }
            .text-gray-800 {
                color: #333333;
            }
            .border-gray-200, .border-gray-300 {
                border-color: #D1D5DB;
            }
            .shadow-2xl {
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25), 0 10px 10px -5px rgba(0, 0, 0, 0.1);
            }
            .bg-fi-chat-user {
                background-color: #E0E0FF; /* Light purple for user messages */
            }
            .bg-fi-chat-artha {
                background-color: #FFFFFF; /* White for Artha messages */
            }
            .chat-message p {
                color: #333333;
            }
            .placeholder-fi-gray::placeholder {
                color: #9CA3AF;
            }
            input {
                background-color: #FFFFFF;
                color: #333333;
            }

            /* Dark mode styles - applied when html has 'dark' class */
            html.dark {
                background-color: var(--fi-dark-background);
                color: var(--fi-text-primary);
            }
            body.dark {
                background-color: var(--fi-dark-background);
                color: var(--fi-text-primary);
            }
            html.dark .bg-white {
                background-color: var(--fi-card-background);
            }
            html.dark .text-gray-800 {
                color: var(--fi-text-primary);
            }
            html.dark .border-gray-200, html.dark .border-gray-300 {
                border-color: var(--fi-border-color);
            }
            html.dark .shadow-2xl {
                box-shadow: 0 25px 50px -12px var(--fi-shadow-color), 0 10px 10px -5px var(--fi-shadow-color);
            }
            /* Adjusted dark mode message backgrounds for better distinction */
            html.dark .bg-fi-chat-user {
                background-color: var(--fi-user-message-bg); /* Charcoal grey for user messages */
                color: var(--fi-text-primary);
            }
            html.dark .bg-fi-chat-artha {
                background-color: var(--fi-artha-message-bg); /* Very dark grey/off-black for Artha messages */
                color: var(--fi-text-primary);
            }
            html.dark .placeholder-fi-gray::placeholder {
                color: var(--fi-text-secondary);
            }
            html.dark input {
                background-color: var(--fi-artha-message-bg);
                color: var(--fi-text-primary);
            }

            /* Accent colors */
            .text-fi-primary { color: var(--fi-accent-purple); }
            .border-fi-accent { border-color: var(--fi-accent-yellow); }

            /* Button Gradient */
            .btn-gradient {
                background: linear-gradient(to right, var(--fi-gradient-start), var(--fi-gradient-end));
            }
            .btn-gradient:hover {
                background: linear-gradient(to right, #5A52D9, #7A1BD1);
            }

            /* General rounded corners and shadows for elements */
            .rounded-xl { border-radius: 0.75rem; }
            .rounded-lg { border-radius: 0.5rem; }
            .shadow-md { box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }
            .shadow-lg { box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); }
            .shadow-inner { box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06); }

            /* Custom styles for the new button layout */
            .suggestion-buttons-container {
                display: flex;
                flex-wrap: wrap; /* Allows buttons to wrap if screen is too small */
                justify-content: space-between; /* Distributes buttons evenly */
                gap: 1rem; /* Space between buttons */
                margin-top: auto; /* Pushes the container to the bottom */
                margin-bottom: 1rem; /* Space above the input field */
                padding: 0 1rem; /* Horizontal padding to match screenshot */
            }

            .suggestion-buttons-container button {
                flex: 1; /* Allows buttons to grow and shrink, taking equal space */
                min-width: 150px; /* Minimum width to prevent them from becoming too small */
                padding: 0.75rem 1.5rem; /* Adjust padding for button size */
                font-size: 0.9rem; /* Slightly smaller font for buttons */
            }

            /* Responsive adjustments for smaller screens */
            @media (max-width: 768px) {
                .suggestion-buttons-container {
                    flex-direction: column; /* Stack buttons vertically on small screens */
                    align-items: stretch; /* Make them full width when stacked */
                }
                .suggestion-buttons-container button {
                    width: 100%; /* Full width when stacked */
                    margin-bottom: 0.5rem; /* Space between stacked buttons */
                }
            }
        </style>
    </head>
    <body class="min-h-screen flex items-center justify-center transition-colors duration-300 ease-in-out">
        <div class="bg-white rounded-xl shadow-2xl p-6 w-screen h-screen flex flex-col">
            <div class="flex justify-between items-center mb-6 border-b-2 border-fi-accent pb-3">
                <h1 class="text-3xl md:text-4xl font-bold text-gray-800">
                    Chat with <span class="text-fi-primary">Artha</span>
                </h1>
                <!-- Dark Mode Toggle -->
                <button id="theme-toggle" class="p-2 rounded-full bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 focus:outline-none transition-colors duration-300">
                    <svg id="moon-icon" class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 118.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path>
                    </svg>
                    <svg id="sun-icon" class="w-6 h-6 hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h1M4 12H3m15.354 5.354l-.707.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"></path>
                    </svg>
                </button>
            </div>

            <div id="chat-box" class="flex-1 overflow-y-auto p-4 border border-gray-200 rounded-lg mb-4 space-y-4 shadow-inner">
                <!-- Chat messages will be appended here -->
                <div class="chat-message artha p-3 rounded-lg max-w-[85%] bg-fi-chat-artha shadow-md">
                    <p>Hello, Tharun. Artha is here. ?</p>
                </div>
            </div>

            <!-- New Suggestion Buttons Container - MOVED TO BOTTOM -->
            <div class="suggestion-buttons-container">
                <button id="btn-home-loan" class="btn-gradient text-white rounded-lg shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-fi-primary focus:ring-opacity-50 transition duration-200 ease-in-out transform hover:scale-105">
                    Home Loan Affordability
                </button>
                <button id="btn-regret-analysis" class="btn-gradient text-white rounded-lg shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-fi-primary focus:ring-opacity-50 transition duration-200 ease-in-out transform hover:scale-105">
                    Analyze Regrets
                </button>
                <button id="btn-spending-patterns" class="btn-gradient text-white rounded-lg shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-fi-primary focus:ring-opacity-50 transition duration-200 ease-in-out transform hover:scale-105">
                    Analyze Spending Patterns
                </button>
            </div>

            <div class="flex items-center space-x-3">
                <input type="text" id="user-input" placeholder="Ask Artha anything about your finances..."
                        class="flex-1 border border-gray-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-fi-primary focus:border-transparent placeholder-fi-gray">
                <!-- New Live Audio Button -->
                <button id="btn-live-audio"
                        class="btn-gradient text-white px-6 py-3 rounded-lg shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-fi-primary focus:ring-opacity-50 transition duration-200 ease-in-out transform hover:scale-105">
                    Live Audio
                </button>
                <button id="send-button"
                        class="btn-gradient text-white px-6 py-3 rounded-lg shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-fi-primary focus:ring-opacity-50 transition duration-200 ease-in-out transform hover:scale-105">
                    Send
                </button>
            </div>
        </div>

        <script>
            const chatBox = document.getElementById('chat-box');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');
            const themeToggle = document.getElementById('theme-toggle');
            const moonIcon = document.getElementById('moon-icon');
            const sunIcon = document.getElementById('sun-icon');

            // New buttons
            const btnHomeLoan = document.getElementById('btn-home-loan');
            const btnRegretAnalysis = document.getElementById('btn-regret-analysis');
            const btnSpendingPatterns = document.getElementById('btn-spending-patterns');
            const btnLiveAudio = document.getElementById('btn-live-audio'); // Get reference to the new button


            // Check for saved theme preference or default to light
            if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                document.documentElement.classList.add('dark'); // Apply to html element
                moonIcon.classList.add('hidden');
                sunIcon.classList.remove('hidden');
            } else {
                document.documentElement.classList.remove('dark'); // Apply to html element
                moonIcon.classList.remove('hidden');
                sunIcon.classList.add('hidden');
            }

            themeToggle.addEventListener('click', () => {
                if (document.documentElement.classList.contains('dark')) {
                    document.documentElement.classList.remove('dark');
                    localStorage.theme = 'light';
                    moonIcon.classList.remove('hidden');
                    sunIcon.classList.add('hidden');
                } else {
                    document.documentElement.classList.add('dark');
                    localStorage.theme = 'dark';
                    moonIcon.classList.add('hidden');
                    sunIcon.classList.remove('hidden');
                }
            });

            function addMessage(sender, message, isHtml = false) {
                const messageDiv = document.createElement('div');
                // Add common classes for styling
                messageDiv.classList.add('chat-message', sender, 'p-3', 'rounded-lg', 'max-w-[85%]', 'shadow-md');

                // Add sender-specific background classes
                if (sender === 'user') {
                    messageDiv.classList.add('ml-auto', 'bg-fi-chat-user'); // Align right and apply user background
                } else {
                    messageDiv.classList.add('bg-fi-chat-artha'); // Apply Artha background
                }

                if (isHtml) {
                    messageDiv.innerHTML = message; // Insert HTML directly
                } else {
                    const pElement = document.createElement('p');
                    pElement.innerHTML = message; // Use innerHTML for basic formatting like line breaks
                    messageDiv.appendChild(pElement);
                }

                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
            }

            async function sendMessage() {
                const query = userInput.value.trim();
                if (query === '') return;

                addMessage('user', query);
                userInput.value = ''; // Clear input

                // Add a typing indicator
                const typingIndicator = document.createElement('div');
                typingIndicator.id = 'typing-indicator';
                // Ensure typing indicator also gets the Artha background class
                typingIndicator.classList.add('chat-message', 'artha', 'p-3', 'rounded-lg', 'max-w-[85%]', 'animate-pulse', 'shadow-md', 'bg-fi-chat-artha');
                typingIndicator.innerHTML = '<p>Artha is thinking...</p>';
                chatBox.appendChild(typingIndicator);
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ query: query })
                    });
                    const data = await response.json();

                    chatBox.removeChild(typingIndicator); // Remove typing indicator

                    if (data.status === 'success') {
                        addMessage('artha', data.response);
                    } else {
                        addMessage('artha', `Artha is having a moment of existential dread. Error: ${data.response}`);
                    }
                } catch (error) {
                    console.error('Error sending message:', error);
                    chatBox.removeChild(typingIndicator); // Remove typing indicator
                    addMessage('artha', 'Artha is currently unreachable. Check the server logs, mortal.');
                }
            }

            // Function to handle analysis button clicks
            async function sendAnalysisRequest(endpoint, buttonText) {
                addMessage('user', buttonText); // Show what user clicked

                const typingIndicator = document.createElement('div');
                typingIndicator.id = 'typing-indicator';
                typingIndicator.classList.add('chat-message', 'artha', 'p-3', 'rounded-lg', 'max-w-[85%]', 'animate-pulse', 'shadow-md', 'bg-fi-chat-artha');
                typingIndicator.innerHTML = '<p>Artha is crunching numbers...</p>';
                chatBox.appendChild(typingIndicator);
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ user_id: "User_A_Financials" }) // Hardcoded user ID
                    });
                    const data = await response.json();

                    chatBox.removeChild(typingIndicator);

                    if (data.status === 'success') {
                        let arthaResponseHtml = `<p>${data.summary.join('<br>')}</p>`; // Join bullet points with line breaks

                        // Add image if chart_file exists
                        if (data.chart_file) {
                            // Assuming chart_file is a relative path to the static image
                            arthaResponseHtml += `<img src="/static/${data.chart_file}" alt="Analysis Chart" class="mt-4 rounded-lg shadow-md max-w-full h-auto">`;
                        }
                        // Add iframe for Sankey diagram if sankey_diagram_file exists
                        if (data.sankey_diagram_file) {
                            // Assuming sankey_diagram_file is a relative path to the static HTML file
                            arthaResponseHtml += `<iframe src="/static/${data.sankey_diagram_file}" class="mt-4 rounded-lg shadow-md w-full" style="height: 400px; border: none;"></iframe>`;
                        }

                        addMessage('artha', arthaResponseHtml, true); // Pass true for isHtml
                    } else {
                        addMessage('artha', `Artha encountered an issue with the ${buttonText.toLowerCase()} analysis. Error: ${data.message || 'Unknown error'}. Details: ${data.details || ''}`);
                        if (data.login_url) {
                            addMessage('artha', `Please login to MCP to proceed: <a href="${data.login_url}" target="_blank" class="text-blue-400 hover:underline">Click here to login</a>`);
                        }
                    }
                } catch (error) {
                    console.error(`Error during ${buttonText.toLowerCase()} analysis:`, error);
                    chatBox.removeChild(typingIndicator);
                    addMessage('artha', `Artha is currently unable to perform the ${buttonText.toLowerCase()} analysis. Check the server logs, mortal.`);
                }
            }

            sendButton.addEventListener('click', sendMessage);
            userInput.addEventListener('keypress', (event) => {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            });

            // Event listener for the new Live Audio button
            btnLiveAudio.addEventListener('click', () => {
                // IMPORTANT: Replace this URL with the actual deployed URL of your AI Studio live audio project.
                window.location.href = 'https://copy-of-artha-live-audio-424591628562.us-west1.run.app/';
            });

            // Event listeners for other suggestion buttons
            btnHomeLoan.addEventListener('click', () => sendAnalysisRequest('/analyze/home_loan', 'Home Loan Affordability'));
            btnRegretAnalysis.addEventListener('click', () => sendAnalysisRequest('/analyze/regret', 'Analyze Regrets'));
            btnSpendingPatterns.addEventListener('click', () => sendAnalysisRequest('/analyze/spending', 'Analyze Spending Patterns'));

        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages, fetches data, and gets Artha's response."""
    user_query = request.json.get('query')
    user_id = "User_A_Financials" # Hardcoded user ID for this example

    if not user_query:
        return jsonify({"status": "error", "response": "No query provided."}), 400

    logging.info(f"User query received: {user_query}")

    # Fetch full financial context
    full_financial_context = get_full_financial_context_from_mcp_shared(user_id)

    if isinstance(full_financial_context, dict) and full_financial_context.get("status") == "error":
        return jsonify({
            "status": "error",
            "response": f"Artha cannot access your financial data. Reason: {full_financial_context.get('message', 'Unknown error')}. Details: {full_financial_context.get('details', '')}"
        }), 500

    # Get Artha's response based on the query and full context
    artha_response = get_artha_response(user_query, full_financial_context)

    return jsonify(artha_response)

# --- New Flask Routes for Analysis Buttons ---
@app.route('/analyze/home_loan', methods=['POST'])
def analyze_home_loan_route():
    user_id = request.json.get('user_id', "User_A_Financials")
    analysis_result = run_home_loan_affordability_analysis(user_id)

    if analysis_result.get("status") == "error":
        return jsonify(analysis_result), 500

    summary_result = summarize_analysis_with_gemini(analysis_result, "Home Loan Affordability")

    response_data = {
        "status": summary_result["status"],
        "summary": summary_result["summary"],
        "chart_file": None, # No chart for this analysis currently
        "sankey_diagram_file": None, # No Sankey for this analysis
        "message": analysis_result.get("message", "") # Include original message if any
    }
    return jsonify(response_data)

@app.route('/analyze/regret', methods=['POST'])
def analyze_regret_route():
    user_id = request.json.get('user_id', "User_A_Financials")
    analysis_result = run_regret_analysis(user_id)

    if analysis_result.get("status") == "error":
        return jsonify(analysis_result), 500

    summary_result = summarize_analysis_with_gemini(analysis_result, "Regret Analysis")

    response_data = {
        "status": summary_result["status"],
        "summary": summary_result["summary"],
        "chart_file": None, # No chart for this analysis currently
        "sankey_diagram_file": None, # No Sankey for this analysis
        "message": analysis_result.get("message", "")
    }
    return jsonify(response_data)


@app.route('/analyze/spending', methods=['POST'])
def analyze_spending_route():
    user_id = request.json.get('user_id', "User_A_Financials")
    analysis_result = run_comprehensive_spending_analysis(user_id)

    if analysis_result.get("status") == "error":
        return jsonify(analysis_result), 500

    summary_result = summarize_analysis_with_gemini(analysis_result, "Spending Patterns Analysis")

    response_data = {
        "status": summary_result["status"],
        "summary": summary_result["summary"],
        "chart_file": analysis_result.get("chart_file"), # Pass chart file if generated
        "sankey_diagram_file": analysis_result.get("sankey_diagram_file"), # Pass Sankey file if generated
        "message": analysis_result.get("message", "")
    }
    return jsonify(response_data)


# Run the Flask app
if __name__ == '__main__':
    # Create a 'static' directory if it doesn't exist for charts
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host='0.0.0.0', port=5000, debug=True)
