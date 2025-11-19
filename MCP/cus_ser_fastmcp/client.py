from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client=MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":["mathserver.py"], ## Ensure correct absolute path
                "transport":"stdio",
            
            },
            "weather": {
                "url": "http://localhost:8000/mcp",  # Ensure server is running here
                "transport": "streamable_http",
            },
            "snowflake":{
                "command":"python",
                "args":["snowflakeServer.py"], ## Ensure correct absolute path
                "transport":"stdio",
            
            }

        }
    )

    import os
    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
    # os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

    tools=await client.get_tools()
    model=ChatGroq(model="qwen/qwen3-32b")
    agent=create_agent(
        model,tools
    )

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )

    print("Math response:", math_response['messages'][-1].content)

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in California?"}]}
    )
    print("Weather response:", weather_response['messages'][-1].content)

    snowflake_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Get me the first 5 rows from the orders table"}]}
    )
    print("Snowflake response:", snowflake_response['messages'][-1].content)

asyncio.run(main())