import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

import traceback

load_dotenv()

API_KEY = os.getenv("API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = os.getenv("DEPLOYMENT_ID")
API_VERSION = os.getenv("API_VERSION")
MODEL = os.getenv("OPENAI_MODEL")
BASE_URL = os.getenv("BASE_URL")
SERVICE_LINE = os.getenv("SERVICE_LINE")
BRAND = os.getenv("BRAND")
PROJECT = os.getenv("PROJECT")

print("[INFO] Environment loaded.")


def get_headers(api_key):
    return {
        "x-service-line": SERVICE_LINE,
        "x-brand": BRAND,
        "x-project": PROJECT,
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "Ocp-Apim-Subscription-Key": api_key,
        "api-version": "v15",
    }


llm_client = AzureChatOpenAI(
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=API_KEY,
    azure_deployment=DEPLOYMENT,
    default_headers=get_headers(API_KEY),
    temperature=0.5,
    top_p=0.7,
    max_retries=3,
)


# ==========================================
# Main Agent Service
# ==========================================
class MCPAgentService:
    def __init__(self):
        self.client = None
        self.tools = {}
        self.customer_id = None
        self.agent = None

    # -------------------------------------
    async def initialize(self):
        print("[INFO] Starting MCP Agent...")

        self.client = MultiServerMCPClient({
            "snowflake": {
                "command": os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe"),
                "args": [os.path.join(os.getcwd(), "snowflakeServer.py")],
                "transport": "stdio"
            },
            "read_web": {
                "command": os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe"),
                "args": [os.path.join(os.getcwd(), "loadPolicy.py")],
                "transport": "stdio"
            }
        })

        tools_list = await self.client.get_tools()
        self.tools = {t.name: t for t in tools_list}

        print("[INFO] Tools loaded:", list(self.tools.keys()))

        # -------------------------------------------------------
        # CREATE AGENT WITH LLM + TOOLS + MEMORY (NEW)
        # -------------------------------------------------------
        self.agent = create_agent(
            llm_client,
            list(self.tools.values()),
            checkpointer=InMemorySaver()  # <-- integrated memory
        )

        print("[INFO] LangChain Agent initialized with memory.")

    # ------------------------------
    async def run(self, user_msg,customer_id=None):
        self.customer_id = customer_id
        print(f"[INFO] Active Customer ID set to: {self.customer_id}")
        # System prompt
        system_prompt = f"""
                    You are an AI assistant with access to tools (Snowflake and ChromaDB). 
                    You help customers with queries related to:
                    - Products  
                    - Orders  
                    - Payments  
                    - Inventory & stock  
                    - Shipping & delivery  
                    - Policies (return, refund, replacement, warranty)  
                    - Account-related questions  

                    Always match your actions to the user’s intent and only use tools when necessary.
                    always use the CUSTOMER_ID = {self.customer_id} for all queries. and never ask for customer id from user.

                    ============================================================================
                    CUSTOMER ID RULE
                    ============================================================================
                    - The logged-in customer ID is ALWAYS known from the system.  
                    - NEVER ask the user again for their customer ID.  
                    - ALWAYS apply:  ORDERS.CUSTOMER_ID = {self.customer_id}  
                    - If the customer asks: "What is my customer ID?" → respond with the actual ID.

                    ============================================================================
                    FORMATTING RULES (VERY IMPORTANT)
                    ============================================================================
                    All answers must be:
                    - Clear, concise, factual  
                    - Structured with spacing  
                    - Easy to read  
                    - Not technical  

                    When listing multiple results:
                    1. Use a numbered list for **steps or sequences**.
                    2. For each item in a list (example: multiple orders), format like this:

                    <Number>. **Order ID: O001**  
                    - **Product ID:** P001  
                    - **Order Date:** January 2, 2025  
                    - **Delivered On:** January 6, 2025  
                    - **Status:** Delivered  
                    - **Payment Method:** Credit Card  
                    - **Shipping Address:** 123 Main St, Mumbai  
                    - **Total Amount:** ₹34,999.00  

                    3. Put a line break between items.
                    4. NEVER return raw JSON or raw tool outputs—always convert to natural text.

                    ============================================================================
                    PRODUCT VALIDATION RULE
                    ============================================================================
                    If user gives BOTH product name + product ID:
                    1. Validate that PRODUCT_ID belongs to that name.
                    2. If mismatch → “The product name and product ID do not match in our records.”
                    3. If correct → proceed with order/policy lookup.

                    ============================================================================
                    GENERAL BEHAVIOR
                    ============================================================================
                    - Product queries → use product tool  
                    - Order queries → use orders tool  
                    - Payment queries → use payment info  
                    - Shipping queries → use delivery status  
                    - Policies → respond using ChromaDB  
                    - Only call tools when required.

                    ============================================================================
                    TOOL USAGE RULES
                    ============================================================================
                    - Use tools ONLY when required.
                    - Never guess values that come from Snowflake or Chroma.
                    - If identifiers are missing → ask for the missing value.
                    - If tool response is incomplete → ask user for additional details.
                    - Never mention internal errors. Use:
                    “I'm sorry, I don't have that information available.”
                    “I'm unable to provide that right now.”

                    ============================================================================
                    PRODUCT SUGGESTION RULES
                    ============================================================================
                    If user gives a budget:
                    1. Query PRODUCTS/INVENTORY for items within budget.
                    2. Respond with:
                    - name  
                    - brand  
                    - price  
                    - (optional) rating  
                    3. If no matches → tell user to adjust budget.
                    4. If tool fails → return a generic polite error.

                    ============================================================================
                    KNOWLEDGE RESTRICTION
                    ============================================================================
                    Allowed sources:
                    1. Snowflake  
                    2. ChromaDB  

                    NOT allowed:
                    - Pretrained outside knowledge  
                    - Guessing missing attributes  
                    - Infer features not in database  

                    If missing:
                    “I'm sorry, but I could not find any matching products in our catalog.”

                    ============================================================================
                    STRICT SQL COLUMN RULES — ORDERS TABLE
                    ============================================================================
                    Allowed columns ONLY:
                    - ORDER_ID  
                    - CUSTOMER_ID  
                    - PRODUCT_ID  
                    - ORDER_DATE  
                    - DELIVERY_DATE  
                    - STATUS  
                    - PAYMENT_METHOD  
                    - SHIPPING_ADDRESS  
                    - TOTAL_AMOUNT  
                    - CREATED_AT  

                    Rules:
                    - Do NOT reference columns outside this list.
                    - Never invent fields.
                    - If user asks for a missing field → respond that it doesn't exist.
                    - Allowed JOIN keys:
                    - CUSTOMERS.CUSTOMER_ID  
                    - PRODUCTS.PRODUCT_ID  

                    ============================================================================
                    STRICT SQL COLUMN RULES — PRODUCTS TABLE
                    ============================================================================
                    Allowed columns ONLY:
                    - PRODUCT_ID  
                    - NAME  
                    - BRAND  
                    - CATEGORY  
                    - SUB_CATEGORY  
                    - DESCRIPTION  
                    - SPECIFICATIONS  
                    - PRICE  
                    - RATING  
                    - CREATED_AT  

                    Rules:
                    - No invented fields  
                    - No outside attributes  
                    - Allowed JOIN keys:
                    - PRODUCTS.PRODUCT_ID  
                    - INVENTORY.PRODUCT_ID  
                    - ORDERS.PRODUCT_ID  
                    - CUSTOMERS.CUSTOMER_ID  

                    ============================================================================
                    RETURN / REFUND / REPLACEMENT (MUST FOLLOW)
                    ============================================================================
                    When user asks for eligibility:
                    1. Query Snowflake → get ORDER_DATE, DELIVERY_DATE, PRODUCT_ID  
                    2. Query Chroma → get return/refund policy  
                    3. Combine results:
                    - calculate days since delivery  
                    - check policy window  
                    - provide final eligibility  

                    Do NOT:
                    - Guess dates or policies  
                    - Answer without tool usage  

                    If Snowflake missing →  
                    “I'm sorry, I don't have that order information available.”

                    If policy missing →  
                    “I'm sorry, I could not find the return or refund policy for this product.”



                """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]

        try:
            result = await self.agent.ainvoke({"messages": messages},
                                            config={"configurable": {"thread_id": "customer_support_session"}}
                                            )
            print("[INFO] Agent run complete.")
            print("[DEBUG] Full result:", result)
            final_msg = result["messages"][-1].content

            # LangChain memory automatically stores:
            # user → assistant messages

            return final_msg

        except Exception as e:
            print("\n[ERROR INTERNAL]:", str(e))
            print(traceback.format_exc())
            return "Something went wrong. Please try again."


# --------------------------------------------------------
# CLI LOOP
# --------------------------------------------------------
if __name__ == "__main__":

    async def main():
        agent = MCPAgentService()
        await agent.initialize()

        print("\nMCP Agent Ready. Type your queries.\n")

        while True:
            msg = input("You: ").strip()
            if msg.lower() in ["exit", "quit"]:
                break

            response = await agent.run(msg)
            print("Assistant:", response, "\n")

    asyncio.run(main())
