from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()


class MCPAgentService:
    def __init__(self):
        self.agent = None
        self.client = None

    async def initialize(self):
        self.client = MultiServerMCPClient(
            {
                "math": {
                    "command": "python",
                    "args": ["mathserver.py"],
                    "transport": "stdio",
                },
                "weather": {
                    "url": "http://localhost:8000/mcp",
                    "transport": "streamable_http",
                },
                "snowflake": {
                    "command": "python",
                    "args": ["snowflakeServer.py"],
                    "transport": "stdio",
                }
            }
        )

        # API keys
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

        tools = await self.client.get_tools()

        model = ChatGroq(model="qwen/qwen3-32b")
        # model = ChatOpenAI(model="gpt-4.1")  # If switching to OpenAI

        self.agent = create_agent(model, tools)

    async def run_query(self, query: str):
        response = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        return response["messages"][-1].content
